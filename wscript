#!/usr/bin/env python
from waflib.extras.test_base import summary

def options(opt):
    opt.load("compiler_cxx")
    opt.load("pytest")
    opt.load("python")


def configure(cfg):
    cfg.load("compiler_cxx")
    cfg.load("pytest")
    cfg.load("python")
    cfg.check_python_version()


def build(bld):
    bld(
        target="nir",
        features="py",
        relative_trick=True,
        source=bld.path.ant_glob("nir/**/*.py"),
        install_from="."
    )

    bld(
        target='nir_tests',
        tests=bld.path.ant_glob('tests/**/*.py'),
        features='use pytest',
        use=['nir'],
        install_path='${PREFIX}/bin/tests',
    )

    # Create test summary (to stdout and XML file)
    bld.add_post_fun(summary)
