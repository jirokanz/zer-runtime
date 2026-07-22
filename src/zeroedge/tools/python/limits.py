
import resource
import sys

def set_resource_limits(cpu_seconds=5.0, memory_mb=128, output_bytes=1048576):
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except ValueError:
        pass
    mem_bytes = memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except ValueError:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))
    except ValueError:
        pass

def enforce_output_limit(output, max_bytes):
    encoded = output.encode("utf-8")
    if len(encoded) > max_bytes:
        return encoded[:max_bytes].decode("utf-8", errors="ignore") + "\n... [TRUNCATED]"
    return output
