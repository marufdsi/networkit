#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include <sys/syscall.h>
#include <linux/perf_event.h>
#ifndef SKYLAKE
#define SKYLAKE true
#endif
#define MSR_RAPL_POWER_UNIT 0x606

/*
 * Platform specific RAPL Domains.
 * Note that PP1 RAPL Domain is supported on 062A only
 * And DRAM RAPL Domain is supported on 062D only
 */
/* Package RAPL Domain */
#define MSR_PKG_RAPL_POWER_LIMIT 0x610
#define MSR_PKG_ENERGY_STATUS 0x611
#define MSR_PKG_PERF_STATUS 0x613
#define MSR_PKG_POWER_INFO 0x614

/* PP0 RAPL Domain */
#define MSR_PP0_POWER_LIMIT 0x638
#define MSR_PP0_ENERGY_STATUS 0x639
#define MSR_PP0_POLICY 0x63A
#define MSR_PP0_PERF_STATUS 0x63B

/* PP1 RAPL Domain, may reflect to uncore devices */
#define MSR_PP1_POWER_LIMIT 0x640
#define MSR_PP1_ENERGY_STATUS 0x641
#define MSR_PP1_POLICY 0x642

/* DRAM RAPL Domain */
#define MSR_DRAM_POWER_LIMIT 0x618
#define MSR_DRAM_ENERGY_STATUS 0x619
#define MSR_DRAM_PERF_STATUS 0x61B
#define MSR_DRAM_POWER_INFO 0x61C

/* PSYS RAPL Domain */
#define MSR_PLATFORM_ENERGY_STATUS 0x64d

/* RAPL UNIT BITMASK */
#define POWER_UNIT_OFFSET 0
#define POWER_UNIT_MASK 0x0F

#define ENERGY_UNIT_OFFSET 0x08
#define ENERGY_UNIT_MASK 0x1F00

#define TIME_UNIT_OFFSET 0x10
#define TIME_UNIT_MASK 0xF000

static int open_msr(int core) {

    char msr_filename[BUFSIZ];
    int fd;

    sprintf(msr_filename, "/dev/cpu/%d/msr", core);
    fd = open(msr_filename, O_RDONLY);
    if (fd < 0) {
        if (errno == ENXIO) {
            fprintf(stderr, "rdmsr: No CPU %d\n", core);
            exit(2);
        } else if (errno == EIO) {
            fprintf(stderr, "rdmsr: CPU %d doesn't support MSRs\n",
                    core);
            exit(3);
        } else {
            perror("rdmsr:open");
            fprintf(stderr, "Trying to open %s\n", msr_filename);
            exit(127);
        }
    }

    return fd;
}

static long long read_msr(int fd, int which) {

    uint64_t data;

    if (pread(fd, &data, sizeof data, which) != sizeof data) {
        perror("rdmsr:pread");
        exit(127);
    }

    return (long long) data;
}

#define CPU_SANDYBRIDGE 42
#define CPU_SANDYBRIDGE_EP 45
#define CPU_IVYBRIDGE 58
#define CPU_IVYBRIDGE_EP 62
#define CPU_HASWELL 60
#define CPU_HASWELL_ULT 69
#define CPU_HASWELL_GT3E 70
#define CPU_HASWELL_EP 63
#define CPU_BROADWELL 61
#define CPU_BROADWELL_GT3E 71
#define CPU_BROADWELL_EP 79
#define CPU_BROADWELL_DE 86
#define CPU_SKYLAKE 78
#define CPU_SKYLAKE_HS 94
#define CPU_SKYLAKE_X 85
#define CPU_KNIGHTS_LANDING 87
#define CPU_KNIGHTS_MILL 133
#define CPU_KABYLAKE_MOBILE 142
#define CPU_KABYLAKE 158
#define CPU_ATOM_SILVERMONT 55
#define CPU_ATOM_AIRMONT 76
#define CPU_ATOM_MERRIFIELD 74
#define CPU_ATOM_MOOREFIELD 90
#define CPU_ATOM_GOLDMONT 92
#define CPU_ATOM_GEMINI_LAKE 122
#define CPU_ATOM_DENVERTON 95

/* TODO: on Skylake, also may support  PSys "platform" domain,*/
/* the whole SoC not just the package.*/
/* see dcee75b3b7f025cc6765e6c92ba0a4e59a4d25f4*/

static int detect_cpu(void) {

    FILE *fff;

    int family, model = -1;
    char buffer[BUFSIZ], *result;
    char vendor[BUFSIZ];

    fff = fopen("/proc/cpuinfo", "r");
    if (fff == NULL) return -1;

    while (1) {
        result = fgets(buffer, BUFSIZ, fff);
        if (result == NULL) break;

        if (!strncmp(result, "vendor_id", 8)) {
            sscanf(result, "%*s%*s%s", vendor);

            if (strncmp(vendor, "GenuineIntel", 12)) {
                printf("%s not an Intel chip\n", vendor);
                return -1;
            }
        }

        if (!strncmp(result, "cpu family", 10)) {
            sscanf(result, "%*s%*s%*s%d", &family);
            if (family != 6) {
                printf("Wrong CPU family %d\n", family);
                return -1;
            }
        }

        if (!strncmp(result, "model", 5)) {
            sscanf(result, "%*s%*s%d", &model);
        }

    }

    fclose(fff);

    printf("Found ");

    switch (model) {
        case CPU_SANDYBRIDGE:
            printf("Sandybridge");
            break;
        case CPU_SANDYBRIDGE_EP:
            printf("Sandybridge-EP");
            break;
        case CPU_IVYBRIDGE:
            printf("Ivybridge");
            break;
        case CPU_IVYBRIDGE_EP:
            printf("Ivybridge-EP");
            break;
        case CPU_HASWELL:
        case CPU_HASWELL_ULT:
        case CPU_HASWELL_GT3E:
            printf("Haswell");
            break;
        case CPU_HASWELL_EP:
            printf("Haswell-EP");
            break;
        case CPU_BROADWELL:
        case CPU_BROADWELL_GT3E:
            printf("Broadwell");
            break;
        case CPU_BROADWELL_EP:
            printf("Broadwell-EP");
            break;
        case CPU_SKYLAKE:
        case CPU_SKYLAKE_HS:
            printf("Skylake");
            break;
        case CPU_SKYLAKE_X:
            printf("Skylake-X");
            break;
        case CPU_KABYLAKE:
        case CPU_KABYLAKE_MOBILE:
            printf("Kaby Lake");
            break;
        case CPU_KNIGHTS_LANDING:
            printf("Knight's Landing");
            break;
        case CPU_KNIGHTS_MILL:
            printf("Knight's Mill");
            break;
        case CPU_ATOM_GOLDMONT:
        case CPU_ATOM_GEMINI_LAKE:
        case CPU_ATOM_DENVERTON:
            printf("Atom");
            break;
        default:
            printf("Unsupported model %d\n", model);
            model = -1;
            break;
    }

    printf(" Processor type\n");

    return model;
}

#define MAX_CPUS 1024
#define MAX_PACKAGES 16

static int total_cores = 0, total_packages = 0;
static int package_map[MAX_PACKAGES];

static int detect_packages(void) {

    char filename[BUFSIZ];
    FILE *fff;
    int package;
    int i;

    for (i = 0; i < MAX_PACKAGES; i++) package_map[i] = -1;

    printf("\t");
    for (i = 0; i < MAX_CPUS; i++) {
        sprintf(filename, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        fff = fopen(filename, "r");
        if (fff == NULL) break;
        fscanf(fff, "%d", &package);
        printf("%d (%d)", i, package);
        if (i % 8 == 7) printf("\n\t"); else printf(", ");
        fclose(fff);

        if (package_map[package] == -1) {
            total_packages++;
            package_map[package] = i;
        }

    }

    printf("\n");

    total_cores = i;

    printf("\tDetected %d cores in %d packages\n\n",
           total_cores, total_packages);

    return 0;
}


/*******************************/
/* MSR code                    */
/*******************************/
static int rapl_msr(int core, int cpu_model) {

    int fd;
    long long result;
    double power_units, time_units;
    double cpu_energy_units[MAX_PACKAGES], dram_energy_units[MAX_PACKAGES];
    double package_before[MAX_PACKAGES], package_after[MAX_PACKAGES];
    double pp0_before[MAX_PACKAGES], pp0_after[MAX_PACKAGES];
    double pp1_before[MAX_PACKAGES], pp1_after[MAX_PACKAGES];
    double dram_before[MAX_PACKAGES], dram_after[MAX_PACKAGES];
    double psys_before[MAX_PACKAGES], psys_after[MAX_PACKAGES];
    double thermal_spec_power, minimum_power, maximum_power, time_window;
    int j;

    int dram_avail = 0, pp0_avail = 0, pp1_avail = 0, psys_avail = 0;
    int different_units = 0;

    printf("\nTrying /dev/msr interface to gather results\n\n");

    if (cpu_model < 0) {
        printf("\tUnsupported CPU model %d\n", cpu_model);
        return -1;
    }

    switch (cpu_model) {

        case CPU_SANDYBRIDGE_EP:
        case CPU_IVYBRIDGE_EP:
            pp0_avail = 1;
            pp1_avail = 0;
            dram_avail = 1;
            different_units = 0;
            psys_avail = 0;
            break;

        case CPU_HASWELL_EP:
        case CPU_BROADWELL_EP:
        case CPU_SKYLAKE_X:
            pp0_avail = 1;
            pp1_avail = 0;
            dram_avail = 1;
            different_units = 1;
            psys_avail = 0;
            break;

        case CPU_KNIGHTS_LANDING:
        case CPU_KNIGHTS_MILL:
            pp0_avail = 0;
            pp1_avail = 0;
            dram_avail = 1;
            different_units = 1;
            psys_avail = 0;
            break;

        case CPU_SANDYBRIDGE:
        case CPU_IVYBRIDGE:
            pp0_avail = 1;
            pp1_avail = 1;
            dram_avail = 0;
            different_units = 0;
            psys_avail = 0;
            break;

        case CPU_HASWELL:
        case CPU_HASWELL_ULT:
        case CPU_HASWELL_GT3E:
        case CPU_BROADWELL:
        case CPU_BROADWELL_GT3E:
        case CPU_ATOM_GOLDMONT:
        case CPU_ATOM_GEMINI_LAKE:
        case CPU_ATOM_DENVERTON:
            pp0_avail = 1;
            pp1_avail = 1;
            dram_avail = 1;
            different_units = 0;
            psys_avail = 0;
            break;

        case CPU_SKYLAKE:
        case CPU_SKYLAKE_HS:
        case CPU_KABYLAKE:
        case CPU_KABYLAKE_MOBILE:
            pp0_avail = 1;
            pp1_avail = 1;
            dram_avail = 1;
            different_units = 0;
            psys_avail = 1;
            break;

    }

    for (j = 0; j < total_packages; j++) {
        printf("\tListing paramaters for package #%d\n", j);

        fd = open_msr(package_map[j]);

        /* Calculate the units used */
        result = read_msr(fd, MSR_RAPL_POWER_UNIT);

        power_units = pow(0.5, (double) (result & 0xf));
        cpu_energy_units[j] = pow(0.5, (double) ((result >> 8) & 0x1f));
        time_units = pow(0.5, (double) ((result >> 16) & 0xf));

        /* On Haswell EP and Knights Landing */
        /* The DRAM units differ from the CPU ones */
        if (different_units) {
            dram_energy_units[j] = pow(0.5, (double) 16);
            printf("DRAM: Using %lf instead of %lf\n",
                   dram_energy_units[j], cpu_energy_units[j]);
        } else {
            dram_energy_units[j] = cpu_energy_units[j];
        }

        printf("\t\tPower units = %.3fW\n", power_units);
        printf("\t\tCPU Energy units = %.8fJ\n", cpu_energy_units[j]);
        printf("\t\tDRAM Energy units = %.8fJ\n", dram_energy_units[j]);
        printf("\t\tTime units = %.8fs\n", time_units);
        printf("\n");

        /* Show package power info */
        result = read_msr(fd, MSR_PKG_POWER_INFO);
        thermal_spec_power = power_units * (double) (result & 0x7fff);
        printf("\t\tPackage thermal spec: %.3fW\n", thermal_spec_power);
        minimum_power = power_units * (double) ((result >> 16) & 0x7fff);
        printf("\t\tPackage minimum power: %.3fW\n", minimum_power);
        maximum_power = power_units * (double) ((result >> 32) & 0x7fff);
        printf("\t\tPackage maximum power: %.3fW\n", maximum_power);
        time_window = time_units * (double) ((result >> 48) & 0x7fff);
        printf("\t\tPackage maximum time window: %.6fs\n", time_window);

        /* Show package power limit */
        result = read_msr(fd, MSR_PKG_RAPL_POWER_LIMIT);
        printf("\t\tPackage power limits are %s\n", (result >> 63) ? "locked" : "unlocked");
        double pkg_power_limit_1 = power_units * (double) ((result >> 0) & 0x7FFF);
        double pkg_time_window_1 = time_units * (double) ((result >> 17) & 0x007F);
        printf("\t\tPackage power limit #1: %.3fW for %.6fs (%s, %s)\n",
               pkg_power_limit_1, pkg_time_window_1,
               (result & (1LL << 15)) ? "enabled" : "disabled",
               (result & (1LL << 16)) ? "clamped" : "not_clamped");
        double pkg_power_limit_2 = power_units * (double) ((result >> 32) & 0x7FFF);
        double pkg_time_window_2 = time_units * (double) ((result >> 49) & 0x007F);
        printf("\t\tPackage power limit #2: %.3fW for %.6fs (%s, %s)\n",
               pkg_power_limit_2, pkg_time_window_2,
               (result & (1LL << 47)) ? "enabled" : "disabled",
               (result & (1LL << 48)) ? "clamped" : "not_clamped");

        /* only available on *Bridge-EP */
        if ((cpu_model == CPU_SANDYBRIDGE_EP) || (cpu_model == CPU_IVYBRIDGE_EP)) {
            result = read_msr(fd, MSR_PKG_PERF_STATUS);
            double acc_pkg_throttled_time = (double) result * time_units;
            printf("\tAccumulated Package Throttled Time : %.6fs\n",
                   acc_pkg_throttled_time);
        }

        /* only available on *Bridge-EP */
        if ((cpu_model == CPU_SANDYBRIDGE_EP) || (cpu_model == CPU_IVYBRIDGE_EP)) {
            result = read_msr(fd, MSR_PP0_PERF_STATUS);
            double acc_pp0_throttled_time = (double) result * time_units;
            printf("\tPowerPlane0 (core) Accumulated Throttled Time "
                   ": %.6fs\n", acc_pp0_throttled_time);

            result = read_msr(fd, MSR_PP0_POLICY);
            int pp0_policy = (int) result & 0x001f;
            printf("\tPowerPlane0 (core) for core %d policy: %d\n", core, pp0_policy);

        }


        if (pp1_avail) {
            result = read_msr(fd, MSR_PP1_POLICY);
            int pp1_policy = (int) result & 0x001f;
            printf("\tPowerPlane1 (on-core GPU if avail) %d policy: %d\n",
                   core, pp1_policy);
        }
        close(fd);

    }
    printf("\n");

    for (j = 0; j < total_packages; j++) {

        fd = open_msr(package_map[j]);

        /* Package Energy */
        result = read_msr(fd, MSR_PKG_ENERGY_STATUS);
        package_before[j] = (double) result * cpu_energy_units[j];

        /* PP0 energy */
        /* Not available on Knights* */
        /* Always returns zero on Haswell-EP? */
        if (pp0_avail) {
            result = read_msr(fd, MSR_PP0_ENERGY_STATUS);
            pp0_before[j] = (double) result * cpu_energy_units[j];
        }

        /* PP1 energy */
        /* not available on *Bridge-EP */
        if (pp1_avail) {
            result = read_msr(fd, MSR_PP1_ENERGY_STATUS);
            pp1_before[j] = (double) result * cpu_energy_units[j];
        }


        /* Updated documentation (but not the Vol3B) says Haswell and*/
        /* Broadwell have DRAM support too*/
        if (dram_avail) {
            result = read_msr(fd, MSR_DRAM_ENERGY_STATUS);
            dram_before[j] = (double) result * dram_energy_units[j];
        }


        /* Skylake and newer for Psys*/
        if ((cpu_model == CPU_SKYLAKE) ||
            (cpu_model == CPU_SKYLAKE_HS) ||
            (cpu_model == CPU_KABYLAKE) ||
            (cpu_model == CPU_KABYLAKE_MOBILE)) {

            result = read_msr(fd, MSR_PLATFORM_ENERGY_STATUS);
            psys_before[j] = (double) result * cpu_energy_units[j];
        }

        close(fd);
    }

    printf("\n\tSleeping 1 second\n\n");
    sleep(1);

    for (j = 0; j < total_packages; j++) {

        fd = open_msr(package_map[j]);

        printf("\tPackage %d:\n", j);

        result = read_msr(fd, MSR_PKG_ENERGY_STATUS);
        package_after[j] = (double) result * cpu_energy_units[j];
        printf("\t\tPackage energy: %.6fJ\n",
               package_after[j] - package_before[j]);

        result = read_msr(fd, MSR_PP0_ENERGY_STATUS);
        pp0_after[j] = (double) result * cpu_energy_units[j];
        printf("\t\tPowerPlane0 (cores): %.6fJ\n",
               pp0_after[j] - pp0_before[j]);

        /* not available on SandyBridge-EP */
        if (pp1_avail) {
            result = read_msr(fd, MSR_PP1_ENERGY_STATUS);
            pp1_after[j] = (double) result * cpu_energy_units[j];
            printf("\t\tPowerPlane1 (on-core GPU if avail): %.6f J\n",
                   pp1_after[j] - pp1_before[j]);
        }

        if (dram_avail) {
            result = read_msr(fd, MSR_DRAM_ENERGY_STATUS);
            dram_after[j] = (double) result * dram_energy_units[j];
            printf("\t\tDRAM: %.6fJ\n",
                   dram_after[j] - dram_before[j]);
        }

        if (psys_avail) {
            result = read_msr(fd, MSR_PLATFORM_ENERGY_STATUS);
            psys_after[j] = (double) result * cpu_energy_units[j];
            printf("\t\tPSYS: %.6fJ\n",
                   psys_after[j] - psys_before[j]);
        }

        close(fd);
    }
    printf("\n");
    printf("Note: the energy measurements can overflow in 60s or so\n");
    printf("      so try to sample the counters more often than that.\n\n");

    return 0;
}

static int perf_event_open(struct perf_event_attr *hw_event_uptr,
                           pid_t pid, int cpu, int group_fd, unsigned long flags) {

    return syscall(__NR_perf_event_open, hw_event_uptr, pid, cpu,
                   group_fd, flags);
}

#define NUM_RAPL_DOMAINS 5

extern char rapl_domain_names[NUM_RAPL_DOMAINS][30];


static int check_paranoid(void) {

    int paranoid_value;
    FILE *fff;

    fff = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
    if (fff == NULL) {
        fprintf(stderr, "Error! could not open /proc/sys/kernel/perf_event_paranoid %s\n",
                strerror(errno));

/* We can't return a negative value as that implies no paranoia */
        return 500;
    }

    fscanf(fff, "%d", &paranoid_value);
    fclose(fff);

    return paranoid_value;

}

static int rapl_perf(int core) {

    FILE *fff;
    int type;
    int config[NUM_RAPL_DOMAINS];
    char units[NUM_RAPL_DOMAINS][BUFSIZ];
    char filename[BUFSIZ];
    int fd[NUM_RAPL_DOMAINS][MAX_PACKAGES];
    double scale[NUM_RAPL_DOMAINS];
    struct perf_event_attr attr;
    long long value;
    int i, j;
    int paranoid_value;

    printf("\nTrying perf_event interface to gather results\n\n");

    fff = fopen("/sys/bus/event_source/devices/power/type", "r");
    if (fff == NULL) {
        printf("\tNo perf_event rapl support found (requires Linux 3.14)\n");
        printf("\tFalling back to raw msr support\n\n");
        return -1;
    }
    fscanf(fff, "%d", &type);
    fclose(fff);

    for (i = 0; i < NUM_RAPL_DOMAINS; i++) {

        sprintf(filename, "/sys/bus/event_source/devices/power/events/%s",
                rapl_domain_names[i]);

        fff = fopen(filename, "r");

        if (fff != NULL) {
            fscanf(fff, "event=%x", &config[i]);
            printf("\tEvent=%s Config=%d ", rapl_domain_names[i], config[i]);
            fclose(fff);
        } else {
            continue;
        }

        sprintf(filename, "/sys/bus/event_source/devices/power/events/%s.scale",
                rapl_domain_names[i]);
        fff = fopen(filename, "r");

        if (fff != NULL) {
            fscanf(fff, "%lf", &scale[i]);
            printf("scale=%g ", scale[i]);
            fclose(fff);
        }

        sprintf(filename, "/sys/bus/event_source/devices/power/events/%s.unit",
                rapl_domain_names[i]);
        fff = fopen(filename, "r");

        if (fff != NULL) {
            fscanf(fff, "%s", units[i]);
            printf("units=%s ", units[i]);
            fclose(fff);
        }

        printf("\n");
    }

    for (j = 0; j < total_packages; j++) {

        for (i = 0; i < NUM_RAPL_DOMAINS; i++) {

            fd[i][j] = -1;

            memset(&attr, 0x0, sizeof(attr));
            attr.type = type;
            attr.config = config[i];
            if (config[i] == 0) continue;

            fd[i][j] = perf_event_open(&attr, -1, package_map[j], -1, 0);
            if (fd[i][j] < 0) {
                if (errno == EACCES) {
                    paranoid_value = check_paranoid();
                    if (paranoid_value > 0) {
                        printf("\t/proc/sys/kernel/perf_event_paranoid is %d\n", paranoid_value);
                        printf("\tThe value must be 0 or lower to read system-wide RAPL values\n");
                    }

                    printf("\tPermission denied; run as root or adjust paranoid value\n\n");
                    return -1;
                } else {
                    printf("\terror opening core %d config %d: %s\n\n",
                           package_map[j], config[i], strerror(errno));
                    return -1;
                }
            }
        }
    }

    printf("\n\tSleeping 1 second\n\n");
    sleep(1);

    for (j = 0; j < total_packages; j++) {
        printf("\tPackage %d:\n", j);

        for (i = 0; i < NUM_RAPL_DOMAINS; i++) {

            if (fd[i][j] != -1) {
                read(fd[i][j], &value, 8);
                close(fd[i][j]);

                printf("\t\t%s Energy Consumed: %lf %s\n",
                       rapl_domain_names[i],
                       (double) value * scale[i],
                       units[i]);

            }

        }

    }
    printf("\n");

    return 0;
}

extern char event_names[MAX_PACKAGES][NUM_RAPL_DOMAINS][256];
extern char filenames[MAX_PACKAGES][NUM_RAPL_DOMAINS][256];
extern char basenames[MAX_PACKAGES][256];
extern char tempfile[256];
extern long long before[MAX_PACKAGES][NUM_RAPL_DOMAINS];
extern long long after[MAX_PACKAGES][NUM_RAPL_DOMAINS];
extern int valid[MAX_PACKAGES][NUM_RAPL_DOMAINS];

extern FILE *rapl_sysf_f;

static int rapl_sysfs_init() {
    total_packages = 0;
    total_cores = 0;
    strcpy (rapl_domain_names[0], "energy-cores");
    strcpy (rapl_domain_names[1], "energy-gpu");
    strcpy (rapl_domain_names[2], "energy-pkg");
    strcpy (rapl_domain_names[3], "energy-ram");
    strcpy (rapl_domain_names[4], "energy-psys");
    int cpu_model = detect_cpu();
    detect_packages();
    int i, j;
    for (j = 0; j < total_packages; j++) {
        i = 0;
        sprintf(basenames[j], "/sys/class/powercap/intel-rapl/intel-rapl:%d",
                j);
        sprintf(tempfile, "%s/name", basenames[j]);
        rapl_sysf_f = fopen(tempfile, "r");
        if (rapl_sysf_f == NULL) {
            fprintf(stderr, "\tCould not open %s\n", tempfile);
            return -1;
        }
        fscanf(rapl_sysf_f, "%s", event_names[j][i]);
        valid[j][i] = 1;
        fclose(rapl_sysf_f);
        sprintf(filenames[j][i], "%s/energy_uj", basenames[j]);

        /* Handle subdomains */
        for (i = 1; i < NUM_RAPL_DOMAINS; i++) {
            sprintf(tempfile, "%s/intel-rapl:%d:%d/name",
                    basenames[j], j, i - 1);
            rapl_sysf_f = fopen(tempfile, "r");
            if (rapl_sysf_f == NULL) {
                //fprintf(stderr,"\tCould not open %s\n",tempfile);
                valid[j][i] = 0;
                continue;
            }
            valid[j][i] = 1;
            fscanf(rapl_sysf_f, "%s", event_names[j][i]);
            fclose(rapl_sysf_f);
            sprintf(filenames[j][i], "%s/intel-rapl:%d:%d/energy_uj",
                    basenames[j], j, i - 1);

        }
    }
    return 0;
}

static void rapl_sysfs_before() {
    int i, j;
    /* Gather before values */
    for (j = 0; j < total_packages; j++) {
        for (i = 0; i < NUM_RAPL_DOMAINS; i++) {
            if (valid[j][i]) {
                rapl_sysf_f = fopen(filenames[j][i], "r");
                if (rapl_sysf_f == NULL) {
                    fprintf(stderr, "\tError opening %s!\n", filenames[j][i]);
                } else {
                    fscanf(rapl_sysf_f, "%lld", &before[j][i]);
                    fclose(rapl_sysf_f);
                }
            }
        }
    }
}

static void rapl_sysfs_after() {
    int i, j;
    /* Gather after values */
    for (j = 0; j < total_packages; j++) {
        for (i = 0; i < NUM_RAPL_DOMAINS; i++) {
            if (valid[j][i]) {
                rapl_sysf_f = fopen(filenames[j][i], "r");
                if (rapl_sysf_f == NULL) {
                    fprintf(stderr, "\tError opening %s!\n", filenames[j][i]);
                } else {
                    fscanf(rapl_sysf_f, "%lld", &after[j][i]);
                    fclose(rapl_sysf_f);
                }
            }
        }
    }

}

static void rapl_sysfs_results(std::string version, std::string graph_name, int threads, int iter, int arch=1) {
    int i, j;
    std::ofstream power_log;

    std::string folderName = "EUPAR_Results/PowerLog/";
    if (mkdir(folderName.c_str(), 0777) == -1)
        std::cout<<"Directory " << folderName << " is already exist" << std::endl;
    else
        std::cout<<"Directory " << folderName << " created" << std::endl;

    std::string logFileName = folderName + "PowerLog_" + (arch == 1 ? "Skylake" : "Cascade") + ".csv";
    std::ifstream fileExist(logFileName);
    bool file_exist = fileExist.good();
    power_log.open(logFileName, std::ios_base::out | std::ios_base::app | std::ios_base::ate);
    if (!file_exist) {
        power_log << "Version" << "," << "Graph Name" << "," << "Threads" << "," << "Event" << "," << "Power" << "," << "Iteration" << std::endl;
    }
    for (j = 0; j < total_packages; j++) {
        printf("\tPackage %d\n", j);
        for (i = 0; i < NUM_RAPL_DOMAINS; i++) {
            if (valid[j][i]) {
                power_log<<version << "," << graph_name<<","<<threads<<","<< event_names[j][i] << ","
                << (((double) after[j][i] - (double) before[j][i]) / 1000000.0) << "," << iter <<std::endl;
                printf("\t\t%s\t: %lfJ\n", event_names[j][i], ((double) after[j][i] - (double) before[j][i]) / 1000000.0);
            }
        }
    }
    printf("\n");
}

static int rapl_sysfs(int core) {

    char rapl_event_names[MAX_PACKAGES][NUM_RAPL_DOMAINS][256];
    char rapl_filenames[MAX_PACKAGES][NUM_RAPL_DOMAINS][256];
    char basename[MAX_PACKAGES][256];
    char rapl_tempfile[256];
    long long rapl_before[MAX_PACKAGES][NUM_RAPL_DOMAINS];
    long long rapl_after[MAX_PACKAGES][NUM_RAPL_DOMAINS];
    int rapl_valid[MAX_PACKAGES][NUM_RAPL_DOMAINS];
    int i, j;
    FILE *fff;

    printf("\nTrying sysfs powercap interface to gather results\n\n");

/* /sys/class/powercap/intel-rapl/intel-rapl:0/ */
/* name has name */
/* energy_uj has energy */
/* subdirectories intel-rapl:0:0 intel-rapl:0:1 intel-rapl:0:2 */

    for (j = 0; j < total_packages; j++) {
        i = 0;
        sprintf(basename[j], "/sys/class/powercap/intel-rapl/intel-rapl:%d",
                j);
        sprintf(rapl_tempfile, "%s/name", basename[j]);
        fff = fopen(rapl_tempfile, "r");
        if (fff == NULL) {
            fprintf(stderr, "\tCould not open %s\n", rapl_tempfile);
            return -1;
        }
        fscanf(fff, "%s", rapl_event_names[j][i]);
        rapl_valid[j][i] = 1;
        fclose(fff);
        sprintf(rapl_filenames[j][i], "%s/energy_uj", basename[j]);

/* Handle subdomains */
        for (i = 1; i < NUM_RAPL_DOMAINS; i++) {
            sprintf(rapl_tempfile, "%s/intel-rapl:%d:%d/name",
                    basename[j], j, i - 1);
            fff = fopen(rapl_tempfile, "r");
            if (fff == NULL) {
//fprintf(stderr,"\tCould not open %s\n",rapl_tempfile);
                rapl_valid[j][i] = 0;
                continue;
            }
            rapl_valid[j][i] = 1;
            fscanf(fff, "%s", rapl_event_names[j][i]);
            fclose(fff);
            sprintf(rapl_filenames[j][i], "%s/intel-rapl:%d:%d/energy_uj",
                    basename[j], j, i - 1);

        }
    }

/* Gather before values */
    for (j = 0; j < total_packages; j++) {
        for (i = 0; i < NUM_RAPL_DOMAINS; i++) {
            if (rapl_valid[j][i]) {
                fff = fopen(rapl_filenames[j][i], "r");
                if (fff == NULL) {
                    fprintf(stderr, "\tError opening %s!\n", rapl_filenames[j][i]);
                } else {
                    fscanf(fff, "%lld", &rapl_before[j][i]);
                    fclose(fff);
                }
            }
        }
    }

    printf("\tSleeping 1 second\n\n");
    sleep(1);

/* Gather after values */
    for (j = 0; j < total_packages; j++) {
        for (i = 0; i < NUM_RAPL_DOMAINS; i++) {
            if (rapl_valid[j][i]) {
                fff = fopen(rapl_filenames[j][i], "r");
                if (fff == NULL) {
                    fprintf(stderr, "\tError opening %s!\n", rapl_filenames[j][i]);
                } else {
                    fscanf(fff, "%lld", &rapl_after[j][i]);
                    fclose(fff);
                }
            }
        }
    }

    for (j = 0; j < total_packages; j++) {
        printf("\tPackage %d\n", j);
        for (i = 0; i < NUM_RAPL_DOMAINS; i++) {
            if (rapl_valid[j][i]) {
                printf("\t\t%s\t: %lfJ\n", rapl_event_names[j][i],
                       ((double) rapl_after[j][i] - (double) rapl_before[j][i]) / 1000000.0);
            }
        }
    }
    printf("\n");

    return 0;

}
