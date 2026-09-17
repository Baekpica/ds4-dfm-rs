#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#define DS4_NO_GPU
#include "../ds4_server.c"

int main(void) {
    test_credit_union_merge_shapes();
    test_credit_union_sums_per_run_need();
    test_credit_union_short_plan();
    if (test_failures) {
        return 1;
    }
    puts("DeepSeek page budget: shared-page union and short-context plan PASS");
    return 0;
}
