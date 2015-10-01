(define-module (pylmm)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (gnu packages)
  #:use-module (gnu packages python)
  #:use-module (guix download)
  #:use-module (guix packages)
  #:use-module (guix git-download)
  #:use-module (guix utils)
  #:use-module (guix build-system gnu)
  #:use-module (guix build-system python)
  #:use-module (guix build-system trivial)
  #:use-module (srfi srfi-1))

(define-public python-pylmm
  (package
    (name "python-pylmm")
    (version "1.0.0")
    (source
     (origin
       (method url-fetch)
       (uri (string-append
             "https://pypi.python.org/packages/source/p/pylmm/pylmm-"
             version ".tar.gz"))
       (sha256
        (base32 "0bzl9f9g34dlhwf09i3fdv7dqqzf2iq0w7d6c2bafx1nla98qfbh"))))
    (build-system python-build-system)
    (arguments '(#:tests? #f))
    (native-inputs
     `(("python-setuptools" ,python-setuptools)))
    (home-page "https://github.com/genenetwork/pylmm_gn2")
    (synopsis "Python LMM resolver")
    (description
      "Python LMM resolver")
    (license license:gpl-3)))

(define-public python2-pylmm
  (package-with-python2 python-pylmm))

