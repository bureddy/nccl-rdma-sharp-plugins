/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef NCCL_DOCA_VERBS_H_
#define NCCL_DOCA_VERBS_H_

#include "config.h"
#include <infiniband/verbs.h>
#include <doca_dev.h>
#include <doca_verbs.h>
#ifdef HAVE_DOCA_VERBS_MP
#include <doca_verbs_mp.h>
#endif
#include "nccl.h"
#include "p2p_plugin.h"

struct ncclIbQp;

/* Initialize the per-device DOCA verbs context (and MRC if NCCL_IB_MRC=1). */
ncclResult_t ncclDocaInitDevice(struct ncclIbDev* ibDev);

/* Allocate the DOCA PD (with refcount on ibDev) and create the per-base CQ. */
ncclResult_t ncclDocaInitBase(struct ncclIbNetCommDevBase* base, struct ncclIbDev* ibDev,
                              void* cq_context, int nqps);
ncclResult_t ncclDocaDestroyBase(struct ncclIbNetCommDevBase* base);

/* Create a DOCA QP and transition it to INIT. Stores rvQp/qpn/qpType into the supplied ncclIbQp. */
ncclResult_t ncclDocaCreateQp(struct ncclIbNetCommDevBase* base, struct ncclIbQp* qp,
                              uint8_t ib_port, void* qp_context, int access_flags);

/* QP state transitions (use ibv_qp_attr to carry the parameters). */
ncclResult_t ncclDocaRtrQp(struct ncclIbQp* qp, struct ncclIbDev* ibDev,
                           struct ibv_qp_attr* qpAttr, uint8_t link_layer);
ncclResult_t ncclDocaRtsQp(struct ncclIbQp* qp, struct ibv_qp_attr* qpAttr);

/* Data path wrappers that translate ibv_*_wr / ibv_wc through doca_verbs_bridge. */
ncclResult_t ncclDocaPostSend(struct doca_verbs_qp* rvQp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr);
ncclResult_t ncclDocaPostRecv(struct doca_verbs_qp* rvQp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr);
ncclResult_t ncclDocaPollCq(struct doca_verbs_cq* rvCq, int numEntries, struct ibv_wc* wcs, int* wrDone);
ncclResult_t ncclDocaDestroyQp(struct doca_verbs_qp* rvQp);

#endif /* NCCL_DOCA_VERBS_H_ */
