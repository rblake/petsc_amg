#!/usr/bin/env python

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.getcwd()+'/config')
  import configure
  PETSC_ARCH = 'debug_std'
  del os.environ['PETSC_DIR']
  del os.environ['PETSC_ARCH']
  configure_options = ['--with-petsc-arch='+PETSC_ARCH,
                       '--download-ml=yes',
                       '-PETSC_ARCH='+PETSC_ARCH,
                       '--download-hypre=yes',
                       '--download-mpich=yes',
                       ]
  configure.petsc_configure(configure_options)


#                       '--download-superlu_dist=yes',
#                       '--download-superlu=yes',
#                       '--download-scalapack=yes',
#                       '--download-blacs=ifneeded',
#                       '--download-mumps=yes',
