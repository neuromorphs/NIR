#!/usr/bin/env python
from waflib.extras.test_base import summary

def options(opt):
    opt.load("compiler_cxx")
    opt.load("test_base")
    opt.load("python")


def configure(cfg):
    cfg.load("compiler_cxx")
    cfg.load("test_base")
    cfg.load("python")
    cfg.check_python_version()


def build(bld):
    bld(
        target="nir",
        features="py",
        relative_trick=True,
        source=bld.path.ant_glob("nir/**/*.py"),
        install_from="nir",
    )

    # Create test summary (to stdout and XML file)
    bld.add_post_fun(summary)
