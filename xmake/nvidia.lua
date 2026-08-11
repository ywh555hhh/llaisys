rule("ivcore.build")
    set_extensions(".cu")
    on_build_file(function (target, sourcefile, opt)
        import("core.project.depend")
        import("utils.progress")

        local objectfile = target:objectfile(sourcefile)
        table.insert(target:objectfiles(), objectfile)
        os.mkdir(path.directory(objectfile))

        local dependfile = target:dependfile(objectfile)
        local argv = {
            "-x", "ivcore",
            "-std=c++17",
            "-fPIC",
            "-O3",
            "-DNDEBUG",
            "-Wno-unknown-pragmas",
            "-Iinclude",
            "-Isrc",
            "-I/usr/local/corex/include",
            "-c", sourcefile,
            "-o", objectfile,
        }

        local dependinfo = target:is_rebuilt() and {} or (depend.load(dependfile) or {})
        if not depend.is_changed(dependinfo, {files = {sourcefile}, values = argv}) then
            return
        end

        progress.show(opt.progress, "${color.build.object}ivcore.$(mode) %s", sourcefile)
        os.vrunv("/usr/local/corex/bin/clang++", argv)

        dependinfo.files = {sourcefile}
        dependinfo.values = argv
        depend.save(dependinfo, dependfile)
    end)
rule_end()

function add_nvidia_support()
    add_defines("ENABLE_NVIDIA_API")
    add_rules("ivcore.build")
    add_includedirs("/usr/local/corex/include", {public = true})
    add_linkdirs("build/ivcore_stub")
    add_linkdirs("/usr/local/corex/lib64")
    add_links("cudart")
    add_rpathdirs("/usr/local/corex/lib64")

    before_build(function (_)
        os.mkdir("build/ivcore_stub")
        os.vrunv("ar", {"rcs", "build/ivcore_stub/libcudadevrt.a"})
    end)

    add_files("src/device/nvidia/nvidia_runtime_api.cu")
    add_files("src/device/nvidia/nvidia_resource.cu")
    add_files("src/ops/add/nvidia/add_nvidia.cu")
    add_files("src/ops/argmax/nvidia/argmax_nvidia.cu")
    add_files("src/ops/embedding/nvidia/embedding_nvidia.cu")
    add_files("src/ops/swiglu/nvidia/swiglu_nvidia.cu")
    add_files("src/ops/rms_norm/nvidia/rms_norm_nvidia.cu")
    add_files("src/ops/rope/nvidia/rope_nvidia.cu")
    add_files("src/ops/self_attention/nvidia/self_attention_nvidia.cu")
    add_files("src/ops/linear/nvidia/linear_nvidia.cu")
end
