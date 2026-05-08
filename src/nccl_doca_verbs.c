/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <config.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <infiniband/verbs.h>
#include <doca_dev.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>
#ifdef HAVE_DOCA_VERBS_MP
#include <doca_verbs_mp.h>
#endif

#include "core.h"
#include "debug.h"
#include "param.h"
#include "p2p_plugin.h"
#include "ibvwrap.h"
#include "nccl_mrc.h"
#include "nccl_doca_verbs.h"

/* These are defined in src/ib_plugin.c via NCCL_PARAM(); they have external linkage. */
int64_t ncclParamIbPkey(void);
int64_t ncclParamIbUseInline(void);
int64_t ncclParamIbQpsPerConn(void);
int64_t ncclParamUseLibMrc(void);

/* Inline data size (matches ncclIbSendFifo size used in libibverbs path). */
#define NCCL_IB_DOCA_INLINE_BYTES 64

#define DOCACHECKGOTO(call, RES, label) do { \
    RES = (call); \
    if (RES != DOCA_SUCCESS) { \
      WARN("%s:%d %s -> %s", __FILE__, __LINE__, #call, doca_error_get_name(RES)); \
      goto label; \
    } \
  } while (0)

static enum doca_mtu_size ibvMtu2Doca(enum ibv_mtu mtu) {
  switch (mtu) {
    case IBV_MTU_256:  return DOCA_MTU_SIZE_256_BYTES;
    case IBV_MTU_512:  return DOCA_MTU_SIZE_512_BYTES;
    case IBV_MTU_1024: return DOCA_MTU_SIZE_1K_BYTES;
    case IBV_MTU_2048: return DOCA_MTU_SIZE_2K_BYTES;
    case IBV_MTU_4096: return DOCA_MTU_SIZE_4K_BYTES;
    default:           return DOCA_MTU_SIZE_4K_BYTES;
  }
}

ncclResult_t ncclDocaInitDevice(struct ncclIbDev* ibDev) {
  doca_error_t docaErr;

  /* Initialize logging + parse hints once (idempotent across devices via static cache). */
  NCCLCHECK(docaMrcInit());

  docaErr = doca_verbs_bridge_verbs_context_create(ibDev->context->device, 0, &ibDev->rvCtx);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: doca_verbs_bridge_verbs_context_create(%s) failed: %s",
         ibDev->devName, doca_error_get_name(docaErr));
    return ncclInternalError;
  }
  docaErr = doca_verbs_query_device(ibDev->rvCtx, &ibDev->verbs_device_attr);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: doca_verbs_query_device(%s) failed: %s", ibDev->devName, doca_error_get_name(docaErr));
    doca_verbs_context_destroy(ibDev->rvCtx);
    ibDev->rvCtx = NULL;
    return ncclInternalError;
  }

#ifdef HAVE_DOCA_VERBS_MP
  if (ncclParamIbMrc()) {
    enum doca_verbs_rx_ooo_psn_win_size psn = (enum doca_verbs_rx_ooo_psn_win_size)0;
    int rcx_type = 0;
    NCCLCHECK(ncclMrcInit(ibDev->context, ibDev->rvCtx, ibDev->verbs_device_attr, &psn, &rcx_type));
    ibDev->psn_win_size = (int)psn;
    ibDev->mrc_rcx_type = rcx_type;
    INFO(NCCL_NET, "NET/IB: MRC enabled on %s (psn_win=%d rcx_type=%d)",
         ibDev->devName, ibDev->psn_win_size, ibDev->mrc_rcx_type);
  }
#else
  if (ncclParamIbMrc()) {
    WARN("NET/IB: NCCL_IB_MRC=1 but plugin was built without doca_verbs_mp.h; ignoring");
  }
#endif

#ifdef HAVE_LIBMRC
  if (ncclParamUseLibMrc()) {
    /* libMrc must share its ibv_ctx with the bridge so that base->pd (a bridge PD)
     * is valid for mrc_create_qp(). The bridge opened a private ibv_ctx via
     * doca_verbs_bridge_verbs_context_create(); fetch it via the bridge accessor. */
    struct ibv_context* bridgeCtx = doca_verbs_bridge_get_ibv_ctx(ibDev->rvCtx);
    if (bridgeCtx == NULL) {
      WARN("NET/IB: doca_verbs_bridge_get_ibv_ctx returned NULL; cannot init libMrc");
      return ncclSystemError;
    }
    ibDev->mrcContext = NULL;
    NCCLCHECK(wrap_mrc_create_context(bridgeCtx, &ibDev->mrcContext));
    INFO(NCCL_NET, "NET/IB: libMrc data path enabled on %s", ibDev->devName);
  }
#else
  if (ncclParamUseLibMrc()) {
    WARN("NET/IB: NCCL_USE_LIBMRC=1 but plugin was built without libnv_mrc; ignoring");
  }
#endif
  return ncclSuccess;
}

static ncclResult_t docaCreateCq(struct doca_verbs_context* rvCtx, void* cq_context,
                                 int cq_size, struct doca_verbs_cq** out_cq) {
  doca_error_t docaErr;
  struct doca_verbs_cq_attr* attr = NULL;

  DOCACHECKGOTO(doca_verbs_cq_attr_create(&attr), docaErr, err);
  DOCACHECKGOTO(doca_verbs_cq_attr_set_cq_context(attr, cq_context), docaErr, err);
  DOCACHECKGOTO(doca_verbs_cq_attr_set_cq_size(attr, cq_size), docaErr, err);
  DOCACHECKGOTO(doca_verbs_cq_attr_set_external_datapath_en(attr, 0), docaErr, err);
  DOCACHECKGOTO(doca_verbs_cq_attr_set_entry_size(attr, DOCA_VERBS_CQ_ENTRY_SIZE_64), docaErr, err);
  DOCACHECKGOTO(doca_verbs_cq_create(rvCtx, attr, out_cq), docaErr, err);
  doca_verbs_cq_attr_destroy(attr);
  return ncclSuccess;
err:
  if (attr) doca_verbs_cq_attr_destroy(attr);
  return ncclInternalError;
}

ncclResult_t ncclDocaInitBase(struct ncclIbNetCommDevBase* base, struct ncclIbDev* ibDev,
                              void* cq_context, int nqps) {
  doca_error_t docaErr;
  ncclResult_t ret;
  int locked = 0;

  pthread_mutex_lock(&ibDev->lock);
  locked = 1;
  if (0 == ibDev->pdRefs++) {
    docaErr = doca_verbs_pd_create(ibDev->rvCtx, &ibDev->rvPd);
    if (docaErr != DOCA_SUCCESS) {
      WARN("NET/IB: doca_verbs_pd_create(%s) failed: %s", ibDev->devName, doca_error_get_name(docaErr));
      ibDev->pdRefs--;
      ret = ncclSystemError; goto unlock;
    }
    ibDev->pd = doca_verbs_bridge_verbs_pd_get_ibv_pd(ibDev->rvPd);
    if (ibDev->pd == NULL) {
      WARN("NET/IB: doca_verbs_bridge_verbs_pd_get_ibv_pd(%s) failed", ibDev->devName);
      doca_verbs_pd_destroy(ibDev->rvPd);
      ibDev->rvPd = NULL;
      ibDev->pdRefs--;
      ret = ncclSystemError; goto unlock;
    }
  }
  base->rvPd = ibDev->rvPd;
  base->pd = ibDev->pd;
  pthread_mutex_unlock(&ibDev->lock);
  locked = 0;

  /* Same sizing rule the libibverbs path used: recv requests can generate 2 completions. */
  int cqSize = 2 * MAX_REQUESTS * (nqps > 0 ? nqps : ncclParamIbQpsPerConn());
  ret = docaCreateCq(ibDev->rvCtx, cq_context, cqSize, &base->rvCq);
  if (ret != ncclSuccess) {
    pthread_mutex_lock(&ibDev->lock);
    if (0 == --ibDev->pdRefs) {
      doca_verbs_pd_destroy(ibDev->rvPd);
      ibDev->rvPd = NULL;
      ibDev->pd = NULL;
    }
    pthread_mutex_unlock(&ibDev->lock);
    return ret;
  }
#ifdef HAVE_LIBMRC
  if (ncclParamUseLibMrc() && ibDev->mrcContext != NULL) {
    NCCLCHECK(wrap_mrc_create_cq(&base->mrcCq, ibDev->mrcContext, cqSize, cq_context, NULL, 0));
  }
#endif
  /* libibverbs CQ used by the flush QP (matches mrc-nccl-plugin's split).
   * Must be created on the bridge's ibv_ctx so it shares a context with base->pd
   * (the bridge opened a private ibv_ctx via context_create); otherwise
   * ibv_create_qp() rejects pd/cq from different contexts with EINVAL. */
  struct ibv_context* bridgeCtx = doca_verbs_bridge_get_ibv_ctx(ibDev->rvCtx);
  if (bridgeCtx == NULL) {
    WARN("NET/IB: doca_verbs_bridge_get_ibv_ctx returned NULL");
    doca_verbs_cq_destroy(base->rvCq);
    base->rvCq = NULL;
    pthread_mutex_lock(&ibDev->lock);
    if (0 == --ibDev->pdRefs) {
      doca_verbs_pd_destroy(ibDev->rvPd);
      ibDev->rvPd = NULL;
      ibDev->pd = NULL;
    }
    pthread_mutex_unlock(&ibDev->lock);
    return ncclSystemError;
  }
  ret = wrap_ibv_create_cq(&base->cq, bridgeCtx, cqSize, cq_context, NULL, 0);
  if (ret != ncclSuccess) {
    doca_verbs_cq_destroy(base->rvCq);
    base->rvCq = NULL;
    pthread_mutex_lock(&ibDev->lock);
    if (0 == --ibDev->pdRefs) {
      doca_verbs_pd_destroy(ibDev->rvPd);
      ibDev->rvPd = NULL;
      ibDev->pd = NULL;
    }
    pthread_mutex_unlock(&ibDev->lock);
    return ret;
  }
  return ncclSuccess;
unlock:
  if (locked) pthread_mutex_unlock(&ibDev->lock);
  return ret;
}

ncclResult_t ncclDocaDestroyBase(struct ncclIbNetCommDevBase* base) {
  doca_error_t docaErr;
#ifdef HAVE_LIBMRC
  if (base->mrcCq) {
    if (wrap_mrc_destroy_cq(base->mrcCq) != ncclSuccess)
      WARN("NET/IB: wrap_mrc_destroy_cq failed");
    base->mrcCq = NULL;
  }
#endif
  if (base->cq) {
    ncclResult_t r = wrap_ibv_destroy_cq(base->cq);
    if (r != ncclSuccess)
      WARN("NET/IB: wrap_ibv_destroy_cq failed");
    base->cq = NULL;
  }
  if (base->rvCq) {
    docaErr = doca_verbs_cq_destroy(base->rvCq);
    if (docaErr != DOCA_SUCCESS)
      WARN("NET/IB: doca_verbs_cq_destroy failed: %s", doca_error_get_name(docaErr));
    base->rvCq = NULL;
  }
  pthread_mutex_lock(&ncclIbDevs[base->ibDevN].lock);
  if (0 == --ncclIbDevs[base->ibDevN].pdRefs) {
    if (ncclIbDevs[base->ibDevN].rvPd != NULL) {
      docaErr = doca_verbs_pd_destroy(ncclIbDevs[base->ibDevN].rvPd);
      if (docaErr != DOCA_SUCCESS)
        WARN("NET/IB: doca_verbs_pd_destroy failed: %s", doca_error_get_name(docaErr));
      ncclIbDevs[base->ibDevN].rvPd = NULL;
      ncclIbDevs[base->ibDevN].pd = NULL;
    }
    if (ncclIbDevs[base->ibDevN].cc_group != NULL) {
      docaErr = doca_verbs_cc_group_destroy(ncclIbDevs[base->ibDevN].cc_group);
      if (docaErr != DOCA_SUCCESS)
        WARN("NET/IB: doca_verbs_cc_group_destroy failed: %s", doca_error_get_name(docaErr));
      ncclIbDevs[base->ibDevN].cc_group = NULL;
    }
  }
  pthread_mutex_unlock(&ncclIbDevs[base->ibDevN].lock);
  base->rvPd = NULL;
  base->pd = NULL;
  return ncclSuccess;
}

ncclResult_t ncclDocaCreateQp(struct ncclIbNetCommDevBase* base, struct ncclIbQp* qp,
                              uint8_t ib_port, void* qp_context, int access_flags, int force_rc) {
  doca_error_t docaErr;
  struct doca_verbs_qp_init_attr* qpInit = NULL;
  struct doca_verbs_qp_attr* qpAttr = NULL;
  struct doca_verbs_qp* rvQp = NULL;
  struct ncclIbDev* ibDev = &ncclIbDevs[base->ibDevN];
  uint32_t qpType;
  int mask;

#ifdef HAVE_DOCA_VERBS_MP
  qpType = (ncclParamIbMrc() && !force_rc) ? DOCA_VERBS_QP_TYPE_RCX : DOCA_VERBS_QP_TYPE_RC;
#else
  qpType = DOCA_VERBS_QP_TYPE_RC;
#endif

  DOCACHECKGOTO(doca_verbs_qp_init_attr_create(&qpInit), docaErr, err);
#ifdef HAVE_DOCA_VERBS_MP
  if (qpType == DOCA_VERBS_QP_TYPE_RCX) {
    DOCACHECKGOTO(doca_verbs_qp_init_attr_set_rcx_type(qpInit, ibDev->mrc_rcx_type), docaErr, err);
  }
#endif
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_pd(qpInit, base->rvPd), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_external_datapath_en(qpInit, 0), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_max_inline_data(qpInit,
                  ncclParamIbUseInline() ? NCCL_IB_DOCA_INLINE_BYTES : 0), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_qp_type(qpInit, qpType), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_qp_context(qpInit, qp_context), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_send_cq(qpInit, base->rvCq), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_sq_wr(qpInit, 2 * MAX_REQUESTS), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_send_max_sges(qpInit, 1), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_sq_sig_all(qpInit, 0), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_receive_cq(qpInit, base->rvCq), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_rq_wr(qpInit, MAX_REQUESTS), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_init_attr_set_receive_max_sges(qpInit, 1), docaErr, err);

  DOCACHECKGOTO(doca_verbs_qp_create(ibDev->rvCtx, qpInit, &rvQp), docaErr, err);
  doca_verbs_qp_init_attr_destroy(qpInit);
  qpInit = NULL;

  /* Transition to INIT. */
  DOCACHECKGOTO(doca_verbs_qp_attr_create(&qpAttr), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_port_num(qpAttr, ib_port), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_pkey_index(qpAttr, ncclParamIbPkey()), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_INIT), docaErr, err);
  mask = DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM
       | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ;
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_write(qpAttr,
                  (access_flags & IBV_ACCESS_REMOTE_WRITE) ? 1 : 0), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_read(qpAttr,
                  (access_flags & IBV_ACCESS_REMOTE_READ) ? 1 : 0), docaErr, err);
  if (qpType != DOCA_VERBS_QP_TYPE_UC) {
    DOCACHECKGOTO(doca_verbs_qp_attr_set_atomic_mode(qpAttr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC), docaErr, err);
    mask |= DOCA_VERBS_QP_ATTR_ATOMIC_MODE;
  }
  DOCACHECKGOTO(doca_verbs_qp_modify(rvQp, qpAttr, mask), docaErr, err);
  doca_verbs_qp_attr_destroy(qpAttr);

  qp->rvQp = rvQp;
  qp->qpn = doca_verbs_qp_get_qpn(rvQp);
  qp->qpType = qpType;
  qp->qp = NULL;

  TRACE(NCCL_NET, "NET/IB: ncclDocaCreateQp port=%u dev=%d devName=%s qpn=%u qpType=%u",
        ib_port, base->ibDevN, ibDev->devName, qp->qpn, qpType);
  return ncclSuccess;
err:
  if (qpInit) doca_verbs_qp_init_attr_destroy(qpInit);
  if (qpAttr) doca_verbs_qp_attr_destroy(qpAttr);
  if (rvQp)   doca_verbs_qp_destroy(rvQp);
  return ncclSystemError;
}

ncclResult_t ncclDocaRtrQp(struct ncclIbQp* qp, struct ncclIbDev* ibDev,
                           struct ibv_qp_attr* qpAttr, uint8_t link_layer) {
  doca_error_t docaErr;
  struct doca_verbs_qp_attr* attr = NULL;
  struct doca_verbs_ah_attr* ah = NULL;
  struct doca_verbs_gid rvGid;
  int mask = DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
             DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
             DOCA_VERBS_QP_ATTR_AH_ATTR;
  if (qp->qpType != DOCA_VERBS_QP_TYPE_UC)
    mask |= DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER | DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC;

  DOCACHECKGOTO(doca_verbs_ah_attr_create(ibDev->rvCtx, &ah), docaErr, err);
  memcpy(rvGid.raw, qpAttr->ah_attr.grh.dgid.raw, sizeof(rvGid.raw));

  if (link_layer == IBV_LINK_LAYER_ETHERNET || qpAttr->ah_attr.is_global) {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IPv4), docaErr, err);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_gid(ah, rvGid), docaErr, err);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_sgid_index(ah, qpAttr->ah_attr.grh.sgid_index), docaErr, err);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_hop_limit(ah, qpAttr->ah_attr.grh.hop_limit), docaErr, err);

    uint32_t tc = qpAttr->ah_attr.grh.traffic_class;
#ifdef HAVE_DOCA_VERBS_MP
    if (qp->qpType == DOCA_VERBS_QP_TYPE_RCX && tc == 0) {
      tc = (ncclParamIbMrcTc() != -1) ? (uint32_t)ncclParamIbMrcTc() : NCCL_IB_MRC_TC_DEFAULT;
    }
#endif
    DOCACHECKGOTO(doca_verbs_ah_attr_set_traffic_class(ah, tc), docaErr, err);
  } else {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_dlid(ah, qpAttr->ah_attr.dlid), docaErr, err);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH), docaErr, err);
  }
  DOCACHECKGOTO(doca_verbs_ah_attr_set_sl(ah, qpAttr->ah_attr.sl), docaErr, err);

  DOCACHECKGOTO(doca_verbs_qp_attr_create(&attr), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_path_mtu(attr, ibvMtu2Doca((enum ibv_mtu)qpAttr->path_mtu)), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rq_psn(attr, qpAttr->rq_psn), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_port_num(attr, qpAttr->ah_attr.port_num), docaErr, err);
  if (qp->qpType != DOCA_VERBS_QP_TYPE_UC) {
    DOCACHECKGOTO(doca_verbs_qp_attr_set_min_rnr_timer(attr, qpAttr->min_rnr_timer), docaErr, err);
    DOCACHECKGOTO(doca_verbs_qp_attr_set_max_dest_rd_atomic(attr, qpAttr->max_dest_rd_atomic), docaErr, err);
  }
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ah_attr(attr, ah), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_dest_qp_num(attr, qpAttr->dest_qp_num), docaErr, err);

#ifdef HAVE_DOCA_VERBS_MP
  if (qp->qpType == DOCA_VERBS_QP_TYPE_RCX) {
    DOCACHECKGOTO(doca_verbs_qp_attr_set_ps_hints(attr, &qp_hints, sizeof(qp_hints)), docaErr, err);
    mask |= DOCA_VERBS_QP_ATTR_PS_HINTS;
    if (ibDev->mrc_rcx_type > 0) {
      DOCACHECKGOTO(doca_verbs_qp_attr_set_rx_ooo_psn_win_size(attr,
                      (enum doca_verbs_rx_ooo_psn_win_size)ibDev->psn_win_size), docaErr, err);
      mask |= DOCA_VERBS_QP_ATTR_RX_OOO_PSN_WIN_SIZE;
    }
    if (ibDev->cc_group == NULL) {
      if (ncclInitCcHints(ibDev->rvCtx, &ibDev->cc_group) != ncclSuccess) {
        WARN("NET/IB: ncclInitCcHints failed");
        goto err;
      }
    }
    if (ibDev->cc_group) {
      DOCACHECKGOTO(doca_verbs_qp_attr_set_cc_group(attr, ibDev->cc_group), docaErr, err);
      mask |= DOCA_VERBS_QP_ATTR_CC_GROUP;
    }
  }
#endif

  DOCACHECKGOTO(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTR), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_modify(qp->rvQp, attr, mask), docaErr, err);
  doca_verbs_qp_attr_destroy(attr);
  doca_verbs_ah_attr_destroy(ah);
  return ncclSuccess;
err:
  if (attr) doca_verbs_qp_attr_destroy(attr);
  if (ah)   doca_verbs_ah_attr_destroy(ah);
  return ncclSystemError;
}

ncclResult_t ncclDocaRtsQp(struct ncclIbQp* qp, struct ibv_qp_attr* qpAttr) {
  doca_error_t docaErr;
  struct doca_verbs_qp_attr* attr = NULL;
  int mask = DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN;

  if (qp->qpType != DOCA_VERBS_QP_TYPE_UC)
    mask |= DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
            DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC;

  DOCACHECKGOTO(doca_verbs_qp_attr_create(&attr), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_sq_psn(attr, qpAttr->sq_psn), docaErr, err);
  if (qp->qpType != DOCA_VERBS_QP_TYPE_UC) {
    DOCACHECKGOTO(doca_verbs_qp_attr_set_ack_timeout(attr, qpAttr->timeout), docaErr, err);
    DOCACHECKGOTO(doca_verbs_qp_attr_set_retry_cnt(attr, qpAttr->retry_cnt), docaErr, err);
    DOCACHECKGOTO(doca_verbs_qp_attr_set_rnr_retry(attr, qpAttr->rnr_retry), docaErr, err);
    DOCACHECKGOTO(doca_verbs_qp_attr_set_max_rd_atomic(attr, qpAttr->max_rd_atomic), docaErr, err);
  }
  DOCACHECKGOTO(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTS), docaErr, err);
  DOCACHECKGOTO(doca_verbs_qp_modify(qp->rvQp, attr, mask), docaErr, err);
  doca_verbs_qp_attr_destroy(attr);
  return ncclSuccess;
err:
  if (attr) doca_verbs_qp_attr_destroy(attr);
  return ncclSystemError;
}

ncclResult_t ncclDocaPostSend(struct doca_verbs_qp* rvQp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) {
  int rc = doca_verbs_bridge_post_send(rvQp, wr, bad_wr);
  if (rc != 0) {
    WARN("NET/IB: doca_verbs_bridge_post_send failed: %d", rc);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclDocaPostRecv(struct doca_verbs_qp* rvQp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr) {
  int rc = doca_verbs_bridge_post_recv(rvQp, wr, bad_wr);
  if (rc != 0) {
    WARN("NET/IB: doca_verbs_bridge_post_recv failed: %d", rc);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclDocaPollCq(struct doca_verbs_cq* rvCq, int numEntries, struct ibv_wc* wcs, int* wrDone) {
  *wrDone = doca_verbs_bridge_poll_cq(rvCq, numEntries, wcs);
  if (*wrDone < 0) {
    WARN("NET/IB: doca_verbs_bridge_poll_cq failed: %d", *wrDone);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclDocaDestroyQp(struct doca_verbs_qp* rvQp) {
  doca_error_t docaErr = doca_verbs_qp_destroy(rvQp);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: doca_verbs_qp_destroy failed: %s", doca_error_get_name(docaErr));
    return ncclSystemError;
  }
  return ncclSuccess;
}
