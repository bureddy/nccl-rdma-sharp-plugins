/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef NCCL_MRC_H_
#define NCCL_MRC_H_

#include "config.h"
#include <stdint.h>
#include <stdbool.h>
#include <doca_dev.h>
#include <doca_log.h>
#include <doca_verbs.h>
#ifdef HAVE_DOCA_VERBS_MP
#include <doca_verbs_mp.h>
#endif
#include "nccl.h"

#define NCCL_IB_MAX_HINTS 5
#define NCCL_IB_MAX_CC_HINTS 4
#define NCCL_IB_MRC_CC_VENDOR_ID 0x1
#define NCCL_IB_MRC_CC_FORMAT_ID 0x1
#define NCCL_IB_MRC_TC_DEFAULT 1
#define DOCA_MRC_PSN_WIN_ARRAY_SIZE 5

struct ncclIbQpHintsLayout {
  uint32_t num_qps_per_peer;
  uint32_t num_send_peers;
  uint32_t group_id;
  uint32_t group_num_qps;
  uint64_t app_id;
};

struct ncclIbCcHints {
  uint8_t version;
  uint8_t reserved[3];
  uint32_t init_rate;
  uint32_t min_rate;
  uint32_t max_rate;
};

extern struct ncclIbQpHintsLayout qp_hints;
extern struct ncclIbCcHints cc_hints;

int64_t ncclParamIbMrc(void);
int64_t ncclParamIbMrcVersion(void);
int64_t ncclParamIbMrcTc(void);
int64_t ncclParamMrcPsnWinSize(void);

ncclResult_t docaMrcInit(void);
ncclResult_t ncclInitCcHints(struct doca_verbs_context* rvCtx, struct doca_verbs_cc_group** cc_group);

#ifdef HAVE_DOCA_VERBS_MP
void ncclMrcSetDefaultHints(void);
bool ncclMrcParseHints(const char* hints_str);
bool ncclMrcParseCcHints(const char* cc_hints_str);
ncclResult_t ncclCheckDocaMrcCaps(struct ibv_device* device,
                                  struct doca_verbs_device_attr* rdma_verbs_device_attr,
                                  int* mrc_rcx_type);
ncclResult_t ncclMrcGetSupportedPsnWinSize(struct doca_verbs_device_attr* doca_attr,
                                           enum doca_verbs_rx_ooo_psn_win_size* psn_win_size);
ncclResult_t ncclMrcInit(struct ibv_context* ibvContext,
                         struct doca_verbs_context* rvCtx,
                         struct doca_verbs_device_attr* verbs_device_attr,
                         enum doca_verbs_rx_ooo_psn_win_size* psn_win_size,
                         int* mrc_rcx_type);
#endif

#endif /* NCCL_MRC_H_ */
