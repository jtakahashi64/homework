#include "kernel.h"
#include "common.h"

extern char __bss[], __bss_end[], __stack_top[];

void kernel_main(void)
{
    printf("\n\nHello %s\n", "World!");
    printf("1 + 2 = %d, %x\n", 1 + 2, 0x1234abcd);

    for (;;)
    {
        __asm__ __volatile__("wfi");
    }
}

__attribute__((section(".text.boot")))
// アセンブリをそのまま出力
__attribute__((naked)) void
boot(void)
{
    __asm__ __volatile__(
        "mv sp, %[stack_top]\n"
        "j kernel_main\n"
        :                              // 出力オペランド
        : [stack_top] "r"(__stack_top) // 入力オペランド
    );
}