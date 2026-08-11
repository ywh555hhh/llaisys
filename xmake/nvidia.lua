function add_nvidia_support()
    add_defines("ENABLE_NVIDIA_API")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart")
    set_values("cuda.rdc", false)
    set_policy("build.cuda.devlink", false)
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC", {force = true})
    end
    add_files("src/device/nvidia/*.cu")
    add_files("src/ops/*/nvidia/*.cu")
end
