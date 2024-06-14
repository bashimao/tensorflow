"""
if_nvpl  is a conditional to check if we are building with NVPL.
"""

_TF_NVPL_ROOT = "TF_NVPL_ROOT"

def if_nvpl(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with NVPL.

    Args:
      if_true: expression to evaluate if building with NVPL.
      if_false: expression to evaluate if building without NVPL.

    Returns:
      a select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        "@org_tensorflow//tensorflow/tsl:linux_x86_64": if_true,
        "//conditions:default": if_false,
    })

def if_enable_nvpl(if_true, if_false = []):
    """Shorthand to select() if we are building with NVPL and NVPL is enabled.

    This is only effective when built with NVPL.

    Args:
      if_true: expression to evaluate if building with NVPL and NVPL is enabled
      if_false: expression to evaluate if building without NVPL or NVPL is not enabled.

    Returns:
      A select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        "@org_tensorflow//tensorflow/tsl/nvpl:enable_nvpl": if_true,
        "//conditions:default": if_false,
    })

def nvpl_deps():
    """Returns the correct set of NVPL library dependencies.

      Shorthand for select() to pull in the correct set of NVPL library deps
      depending on the platform.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return select({
        "@org_tensorflow//tensorflow/tsl/nvpl:enable_nvpl": ["@nvpl//:nvpl"],
        "//conditions:default": [],
    })

def _enable_local_nvpl(repository_ctx):
    return _TF_NVPL_ROOT in repository_ctx.os.environ

def _nvpl_autoconf_impl(repository_ctx):
    """Implementation of the local_nvpl_autoconf repository rule."""

    if _enable_local_nvpl(repository_ctx):
        # Symlink lib and include local folders.
        nvpl_root = repository_ctx.os.environ[_TF_NVPL_ROOT]
        nvpl_lib_path = "%s/lib" % nvpl_root
        repository_ctx.symlink(nvpl_lib_path, "lib")
        nvpl_include_path = "%s/include" % nvpl_root
        repository_ctx.symlink(nvpl_include_path, "include")
        nvpl_license_path = "%s/license.txt" % nvpl_root
        repository_ctx.symlink(nvpl_license_path, "license.txt")
    else:
        # setup remote MVPL repository.
        repository_ctx.download_and_extract(
            repository_ctx.attr.urls,
            sha256 = repository_ctx.attr.sha256,
            stripPrefix = repository_ctx.attr.strip_prefix,
        )

    # Also setup BUILD file.
    repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD")

nvpl_repository = repository_rule(
    implementation = _nvpl_autoconf_impl,
    environ = [
        _TF_NVPL_ROOT,
    ],
    attrs = {
        "build_file": attr.label(),
        "urls": attr.string_list(default = []),
        "sha256": attr.string(default = ""),
        "strip_prefix": attr.string(default = ""),
    },
)
