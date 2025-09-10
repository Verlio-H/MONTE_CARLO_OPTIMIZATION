#pragma once

#include <cstdint>

#define AMX_NOP_OP_IMM5(op, imm5) \
__asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
            __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define AMX_LDX(gpr)    AMX_OP_GPR( 0, gpr)
#define AMX_LDY(gpr)    AMX_OP_GPR( 1, gpr)
#define AMX_STX(gpr)    AMX_OP_GPR( 2, gpr)
#define AMX_STY(gpr)    AMX_OP_GPR( 3, gpr)
#define AMX_LDZ(gpr)    AMX_OP_GPR( 4, gpr)
#define AMX_STZ(gpr)    AMX_OP_GPR( 5, gpr)
#define AMX_LDZI(gpr)   AMX_OP_GPR( 6, gpr)
#define AMX_STZI(gpr)   AMX_OP_GPR( 7, gpr)
#define AMX_EXTRX(gpr)  AMX_OP_GPR( 8, gpr)
#define AMX_EXTRY(gpr)  AMX_OP_GPR( 9, gpr)
#define AMX_FMA64(gpr)  AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr)  AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr)  AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr)  AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr)  AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr)  AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr)  AMX_OP_GPR(16, gpr)
#define AMX_SET()       AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()       AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(gpr) AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr)  AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr) AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr)  AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr) AMX_OP_GPR(22, gpr)

#define PTR_ROW_FLAGS(ptr, row, flags) (((uint64_t)(ptr)) + (((uint64_t)((row) + (flags) * 64)) << 56))

typedef uint8_t amx_reg;

inline static void amx_load512_x(amx_reg row, void *ptr) {
    AMX_LDX(PTR_ROW_FLAGS(ptr, row, 0));
}

inline static void amx_load1024_x(amx_reg row, void *ptr) {
    AMX_LDX(PTR_ROW_FLAGS(ptr, row, 1));
}

inline static void amx_load2048_x(amx_reg row, void *ptr, bool flag_M2) {
    if (flag_M2) {
        AMX_LDX(PTR_ROW_FLAGS(ptr, row + 32, 1));
    } else {
        amx_load1024_x(row, ptr);
        amx_load1024_x(row, (uint8_t *)ptr + 128);
    }
}

inline static void amx_load512_y(amx_reg row, void *ptr) {
    AMX_LDY(PTR_ROW_FLAGS(ptr, row, 0));
}

inline static void amx_load1024_y(amx_reg row, void *ptr) {
    AMX_LDY(PTR_ROW_FLAGS(ptr, row, 1));
}

inline static void amx_load2048_y(amx_reg row, void *ptr, bool flag_M2) {
    if (flag_M2) {
        AMX_LDY(PTR_ROW_FLAGS(ptr, row + 32, 1));
    } else {
        amx_load1024_y(row, ptr);
        amx_load1024_y(row, (uint8_t *)ptr + 128);
    }
}

inline static void amx_load512_z(amx_reg row, void *ptr) {
    AMX_LDZ(PTR_ROW_FLAGS(ptr, row, 0));
}

inline static void amx_load1024_z(amx_reg row, void *ptr) {
    AMX_LDZ(PTR_ROW_FLAGS(ptr, row, 1));
}

inline static void amx_store512_x(amx_reg row, void *ptr) {
    AMX_STX(PTR_ROW_FLAGS(ptr, row, 0));
}

inline static void amx_store1024_x(amx_reg row, void *ptr) {
    AMX_STX(PTR_ROW_FLAGS(ptr, row, 1));
}

inline static void amx_store512_y(amx_reg row, void *ptr) {
    AMX_STY(PTR_ROW_FLAGS(ptr, row, 0));
}

inline static void amx_store1024_y(amx_reg row, void *ptr) {
    AMX_STY(PTR_ROW_FLAGS(ptr, row, 1));
}

inline static void amx_store512_z(amx_reg row, void *ptr) {
    AMX_STZ(PTR_ROW_FLAGS(ptr, row, 0));
}

inline static void amx_store1024_z(amx_reg row, void *ptr) {
    AMX_STZ(PTR_ROW_FLAGS(ptr, row, 1));
}

inline static void amx_vfma64(amx_reg destz, amx_reg srcx, amx_reg srcy, bool skipx, bool skipy, bool skipz) {
    AMX_FMA64(((uint64_t)1 << 63) + ((uint64_t)skipx << 29)
            + ((uint64_t)skipy << 28) + ((uint64_t)skipz << 27)
            + ((uint64_t)destz << 20) + ((uint64_t)srcx << 16)
            + ((uint64_t)srcy << 6));
}

inline static void amx_vfma32_old(amx_reg destz, amx_reg srcx, amx_reg srcy, bool skipx, bool skipy, bool skipz) {
    AMX_FMA32(((uint64_t)1 << 63) + ((uint64_t)skipx << 29)
            + ((uint64_t)skipy << 28) + ((uint64_t)skipz << 27)
            + ((uint64_t)destz << 20) + ((uint64_t)srcx << 16)
            + ((uint64_t)srcy << 6));
}

inline static void amx_vfma32(amx_reg destz, amx_reg srcx, amx_reg srcy) {
//    amx_vfma32_old(destz, srcx, srcy, 0, 0, 0);
    AMX_VECFP(((uint64_t)4 << 42) + (destz << 20) + (srcx << 16) + (srcy << 6));
}

inline static void amx_vfms32(amx_reg destz, amx_reg srcx, amx_reg srcy, bool skipx, bool skipy, bool skipz) {
    AMX_FMS32(((uint64_t)1 << 63) + ((uint64_t)skipx << 29)
            + ((uint64_t)skipy << 28) + ((uint64_t)skipz << 27)
            + ((uint64_t)destz << 20) + ((uint64_t)srcx << 16)
            + ((uint64_t)srcy << 6));
    
}

inline static void amx_fmax32(amx_reg destz, amx_reg srcx) {
    AMX_VECFP(((uint64_t)7 << 47) + ((uint64_t)4 << 42) + (destz << 20) + (srcx << 16));
}

inline static void amx_movyx(amx_reg desty, amx_reg srcx) {
    AMX_EXTRY((1 << 27) + (srcx << 20) + (desty << 6));
}

inline static void amx_movxz(amx_reg destx, amx_reg srcz) {
    AMX_EXTRX((srcz << 20) + (destx << 16));
}

inline static void amx_movyz(amx_reg desty, amx_reg srcz) {
    AMX_EXTRX((1 << 26) + (srcz << 20) + (1 << 10) + (desty << 6));
}

inline static void amx_begin() {
    AMX_SET();
}

inline static void amx_end() {
    AMX_CLR();
}
