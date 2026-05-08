#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See file LICENSE for terms.
#

lib_LTLIBRARIES     = libnccl-net.la

libnccl_net_la_CPPFLAGS = -I$(top_srcdir)/include $(DOCA_VERBS_CPPFLAGS) $(LIBMRC_CPPFLAGS)
libnccl_net_la_CFLAGS   = $(CFLAGS) -DGNU_SOURCE
libnccl_net_la_LIBADD   = -lcudart_static $(LIBMRC_LIBS)
libnccl_net_la_LDFLAGS  = $(LDFLAGS) $(DOCA_VERBS_LDFLAGS) $(LIBMRC_LDFLAGS)

libnccl_net_la_SOURCES = \
	ibvwrap.c \
	utils.c	\
        param.c \
	socket.c \
	p2p_plugin.c \
	ib_plugin.c \
	nccl_doca_verbs.c \
	mrc.c

if BUILD_MRC
NCCL_NET_SYMLINK = libnccl-net-mrc.so
else
NCCL_NET_SYMLINK = libnccl-net-ibext.so
endif

install-exec-hook:
	cd $(libdir) && ln -sf libnccl-net.so $(NCCL_NET_SYMLINK)
