/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <config.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <endian.h>
#include <infiniband/verbs.h>

#include "core.h"
#include "debug.h"
#include "utils.h"
#include "param.h"
#include "nccl_mrc.h"

struct ncclIbQpHintsLayout qp_hints = {0};
struct ncclIbCcHints cc_hints = {0};
static struct doca_log_backend* docaLog;

NCCL_PARAM(IbMrc, "IB_MRC", 0)
NCCL_PARAM(IbMrcVersion, "IB_MRC_VERSION", -1)
NCCL_PARAM(IbMrcTc, "IB_MRC_TC", -1)
NCCL_PARAM(MrcPsnWinSize, "IB_MRC_PSN_WIN_SIZE", 512)

ncclResult_t docaMrcInit(void) {
  doca_error_t docaErr;
  const char* docaEnv = getenv("DOCA_LOG_LEVEL");
  if (docaEnv) {
    uint32_t docaLogLevel = DOCA_LOG_LEVEL_INFO;
    if (!strcasecmp(docaEnv, "debug"))                                         docaLogLevel = DOCA_LOG_LEVEL_DEBUG;
    else if (!strcasecmp(docaEnv, "info"))                                     docaLogLevel = DOCA_LOG_LEVEL_INFO;
    else if (!strcasecmp(docaEnv, "warning") || !strcasecmp(docaEnv, "warn"))  docaLogLevel = DOCA_LOG_LEVEL_WARNING;
    else if (!strcasecmp(docaEnv, "error"))                                    docaLogLevel = DOCA_LOG_LEVEL_ERROR;
    else if (!strcasecmp(docaEnv, "crit") || !strcasecmp(docaEnv, "critical")) docaLogLevel = DOCA_LOG_LEVEL_CRIT;
    else if (!strcasecmp(docaEnv, "disable"))                                  docaLogLevel = DOCA_LOG_LEVEL_DISABLE;

    docaErr = doca_log_backend_create_with_file_sdk(stdout, &docaLog);
    if (docaErr != DOCA_SUCCESS) {
      WARN("NET/IB: doca log backend create failed: %s", doca_error_get_name(docaErr));
      return ncclInternalError;
    }
    docaErr = doca_log_backend_set_sdk_level(docaLog, docaLogLevel);
    if (docaErr != DOCA_SUCCESS) {
      WARN("NET/IB: doca log backend set level failed: %s", doca_error_get_name(docaErr));
      return ncclInternalError;
    }
  }

#ifdef HAVE_DOCA_VERBS_MP
  ncclMrcSetDefaultHints();
  const char* hints = ncclGetEnv("NCCL_IB_HINTS");
  if (hints != NULL && !ncclMrcParseHints(hints)) {
    WARN("NET/IB : Failed to parse NCCL_IB_HINTS, falling back to defaults");
    ncclMrcSetDefaultHints();
  }
  const char* cc_hints_env = ncclGetEnv("NCCL_IB_CC_HINTS");
  if (cc_hints_env != NULL && !ncclMrcParseCcHints(cc_hints_env)) {
    WARN("NET/IB : Failed to parse NCCL_IB_CC_HINTS, falling back to defaults");
  }
#endif

  return ncclSuccess;
}

#ifdef HAVE_DOCA_VERBS_MP

void ncclMrcSetDefaultHints(void) {
  qp_hints.num_qps_per_peer = 1;
  qp_hints.num_send_peers = 1;
  qp_hints.group_id = 1;
  qp_hints.group_num_qps = 128;
  qp_hints.app_id = 0xDEADBEEF;
}

bool ncclMrcParseHints(const char* hints_str) {
  int parsed = 0;
  char buff[256];
  char* saveptr = NULL;
  int arr[NCCL_IB_MAX_HINTS];

  snprintf(buff, sizeof(buff), "%s", hints_str);
  for (char* tok = strtok_r(buff, ",", &saveptr); tok && parsed < NCCL_IB_MAX_HINTS;
       tok = strtok_r(NULL, ",", &saveptr)) {
    arr[parsed++] = atoi(tok);
  }
  if (parsed != NCCL_IB_MAX_HINTS) return false;

  qp_hints.num_qps_per_peer = arr[0];
  qp_hints.num_send_peers   = arr[1];
  qp_hints.group_id         = arr[2];
  qp_hints.group_num_qps    = arr[3];
  qp_hints.app_id           = arr[4];
  return true;
}

bool ncclMrcParseCcHints(const char* cc_hints_str) {
  int parsed = 0;
  char buff[256];
  char* saveptr = NULL;
  int arr[NCCL_IB_MAX_CC_HINTS];

  snprintf(buff, sizeof(buff), "%s", cc_hints_str);
  for (char* tok = strtok_r(buff, ",", &saveptr); tok && parsed < NCCL_IB_MAX_CC_HINTS;
       tok = strtok_r(NULL, ",", &saveptr)) {
    arr[parsed++] = atoi(tok);
  }
  if (parsed != NCCL_IB_MAX_CC_HINTS) return false;

  cc_hints.version   = arr[0];
  cc_hints.init_rate = arr[1];
  cc_hints.min_rate  = arr[2];
  cc_hints.max_rate  = arr[3];
  return true;
}

ncclResult_t ncclCheckDocaMrcCaps(struct ibv_device* device,
                                  struct doca_verbs_device_attr* rdma_verbs_device_attr,
                                  int* mrc_rcx_type) {
  doca_error_t docaErr, rcx0_err, rcx1_err;
  ncclResult_t ret = ncclSuccess;
  size_t max_hints_size = 0;
  struct doca_verbs_device_advanced_transport_attr* advanced_transport_attr = NULL;

  docaErr = doca_verbs_device_advanced_transport_attr_create(&advanced_transport_attr);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: doca advanced transport attr create failed for %s: %s",
         device->name, doca_error_get_name(docaErr));
    return ncclInternalError;
  }
  docaErr = doca_verbs_device_attr_extract_device_advanced_transport_attr(rdma_verbs_device_attr, advanced_transport_attr);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: extract advanced transport attr failed for %s: %s",
         device->name, doca_error_get_name(docaErr));
    ret = ncclInternalError; goto exit;
  }
  docaErr = doca_verbs_device_attr_get_is_qp_type_supported(rdma_verbs_device_attr, DOCA_VERBS_QP_TYPE_RCX);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: RCX QP type not supported on %s: %s", device->name, doca_error_get_name(docaErr));
    ret = ncclInternalError; goto exit;
  }
  rcx0_err = doca_verbs_device_advanced_transport_attr_get_is_rcx_supported(advanced_transport_attr, 0);
  rcx1_err = doca_verbs_device_advanced_transport_attr_get_is_rcx_supported(advanced_transport_attr, 1);
  if (rcx0_err != DOCA_SUCCESS && rcx1_err != DOCA_SUCCESS) {
    WARN("NET/IB: RCX type 0 and type 1 unsupported on %s (err0=%s err1=%s)",
         device->name, doca_error_get_name(rcx0_err), doca_error_get_name(rcx1_err));
    ret = ncclInternalError; goto exit;
  }
  {
    int requested = (int)ncclParamIbMrcVersion();
    if (requested != -1) {
      if (requested != 0 && requested != 1) {
        WARN("NET/IB: invalid NCCL_IB_MRC_VERSION=%d (allowed: 0, 1, -1)", requested);
        ret = ncclInternalError; goto exit;
      }
      doca_error_t need = (requested == 0) ? rcx0_err : rcx1_err;
      if (need != DOCA_SUCCESS) {
        WARN("NET/IB: RCX type %d required for MRC version %d but unsupported on %s: %s",
             requested, requested, device->name, doca_error_get_name(need));
        ret = ncclInternalError; goto exit;
      }
    }
    if (mrc_rcx_type != NULL) {
      *mrc_rcx_type = (requested == -1) ? ((rcx1_err == DOCA_SUCCESS) ? 1 : 0) : requested;
    }
  }
  INFO(NCCL_NET, "NET/IB: doca device %s rcx type 0:%s type 1:%s", device->name,
       (rcx0_err == DOCA_SUCCESS) ? "supported" : "not supported",
       (rcx1_err == DOCA_SUCCESS) ? "supported" : "not supported");
  docaErr = doca_verbs_device_advanced_transport_attr_get_is_ps_hints_supported(advanced_transport_attr);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: doca ps hints not supported");
    ret = ncclInternalError; goto exit;
  }
  max_hints_size = doca_verbs_device_advanced_transport_attr_get_ps_hints_max_size(advanced_transport_attr);
  if (max_hints_size < sizeof(struct ncclIbQpHintsLayout)) {
    WARN("NET/IB: doca ps hints max size %zu < required %zu", max_hints_size, sizeof(struct ncclIbQpHintsLayout));
    ret = ncclInternalError; goto exit;
  }
  docaErr = doca_verbs_device_attr_get_is_cc_group_supported(rdma_verbs_device_attr);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: CC group not supported on %s: %s", device->name, doca_error_get_name(docaErr));
    ret = ncclInternalError; goto exit;
  }
exit:
  doca_verbs_device_advanced_transport_attr_destroy(advanced_transport_attr);
  return ret;
}

ncclResult_t ncclMrcGetSupportedPsnWinSize(struct doca_verbs_device_attr* doca_attr,
                                           enum doca_verbs_rx_ooo_psn_win_size* psn_win_size) {
  static const struct {
    int size;
    enum doca_verbs_rx_ooo_psn_win_size psn_win_size;
  } size_map[DOCA_MRC_PSN_WIN_ARRAY_SIZE] = {
    {512,  DOCA_VERBS_RX_OOO_PSN_WIN_SIZE_512},
    {1024, DOCA_VERBS_RX_OOO_PSN_WIN_SIZE_1K},
    {2048, DOCA_VERBS_RX_OOO_PSN_WIN_SIZE_2K},
    {4096, DOCA_VERBS_RX_OOO_PSN_WIN_SIZE_4K},
    {8192, DOCA_VERBS_RX_OOO_PSN_WIN_SIZE_8K},
  };
  int requested = (int)ncclParamMrcPsnWinSize();
  for (int i = 0; i < DOCA_MRC_PSN_WIN_ARRAY_SIZE; i++) {
    if (requested <= size_map[i].size &&
        doca_verbs_device_attr_get_is_rx_ooo_psn_win_size_supported(doca_attr, size_map[i].psn_win_size) == DOCA_SUCCESS) {
      *psn_win_size = size_map[i].psn_win_size;
      return ncclSuccess;
    }
  }
  WARN("NET/IB: NCCL_IB_MRC_PSN_WIN_SIZE=%d, no supported size found", requested);
  return ncclInternalError;
}

ncclResult_t ncclMrcInit(struct ibv_context* ibvContext,
                         struct doca_verbs_context* rvCtx,
                         struct doca_verbs_device_attr* verbs_device_attr,
                         enum doca_verbs_rx_ooo_psn_win_size* psn_win_size,
                         int* mrc_rcx_type) {
  if (rvCtx == NULL || verbs_device_attr == NULL) {
    WARN("NET/IB: ncclMrcInit called without DOCA verbs context");
    return ncclInvalidArgument;
  }
  NCCLCHECK(ncclCheckDocaMrcCaps(ibvContext->device, verbs_device_attr, mrc_rcx_type));
  NCCLCHECK(ncclMrcGetSupportedPsnWinSize(verbs_device_attr, psn_win_size));
  return ncclSuccess;
}

#endif /* HAVE_DOCA_VERBS_MP */

static ncclResult_t ncclInitCcCaps(struct doca_verbs_context* rvCtx) {
  doca_error_t docaErr;
  ncclResult_t ret = ncclSuccess;
  uint32_t vendor_id = 0, format_id = 0;
  struct doca_verbs_cc_group_caps* caps = NULL;

  docaErr = doca_verbs_query_cc_group_caps(rvCtx, &caps);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: doca cc group caps query failed: %s", doca_error_get_name(docaErr));
    return ncclInternalError;
  }
  vendor_id = doca_verbs_cc_group_caps_get_vendor_id(caps);
  format_id = doca_verbs_cc_group_caps_get_format_id(caps);
  if (vendor_id != NCCL_IB_MRC_CC_VENDOR_ID) {
    WARN("NET/IB: doca cc vendor id %u not supported", vendor_id); ret = ncclInternalError; goto exit;
  }
  if (format_id != NCCL_IB_MRC_CC_FORMAT_ID) {
    WARN("NET/IB: doca cc format id %u not supported", format_id); ret = ncclInternalError; goto exit;
  }
exit:
  doca_verbs_cc_group_caps_free(caps);
  return ret;
}

ncclResult_t ncclInitCcHints(struct doca_verbs_context* rvCtx, struct doca_verbs_cc_group** cc_group) {
  doca_error_t docaErr;
  struct doca_verbs_cc_group_attr* attr = NULL;

  if (cc_hints.version == 0) {
    *cc_group = NULL;
    return ncclSuccess;
  }
  if (ncclInitCcCaps(rvCtx) != ncclSuccess) return ncclInternalError;

  docaErr = doca_verbs_cc_group_attr_create(&attr);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: doca cc group attr create failed: %s", doca_error_get_name(docaErr));
    return ncclInternalError;
  }
  docaErr = doca_verbs_cc_group_attr_set_hint(attr, &cc_hints, sizeof(cc_hints));
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: doca cc group set hint failed: %s", doca_error_get_name(docaErr)); goto err;
  }
  docaErr = doca_verbs_cc_group_create(rvCtx, attr, cc_group);
  if (docaErr != DOCA_SUCCESS) {
    WARN("NET/IB: doca cc group create failed: %s", doca_error_get_name(docaErr)); goto err;
  }
  INFO(NCCL_NET, "NET/IB: CC group ver=%u init_rate=%u min_rate=%u max_rate=%u",
       cc_hints.version, be32toh(cc_hints.init_rate), be32toh(cc_hints.min_rate), be32toh(cc_hints.max_rate));
  doca_verbs_cc_group_attr_destroy(attr);
  return ncclSuccess;
err:
  if (attr) doca_verbs_cc_group_attr_destroy(attr);
  return ncclInternalError;
}
