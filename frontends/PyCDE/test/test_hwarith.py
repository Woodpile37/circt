# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Input, Output, generator
from pycde.testing import unittestmodule

from pycde.dialects import comb

from pycde.pycde_types import dim, types


@unittestmodule(run_passes=False)
class InfixOperators:
  in0 = Input(types.si16)
  in1 = Input(types.ui16)

  @generator
  def construct(ports):
    add = ports.in0 + ports.in1
    sub = ports.in0 - ports.in1
    mul = ports.in0 * ports.in1
    div = ports.in0 / ports.in1
    and_ = ports.in0 & ports.in1
    or_ = ports.in0 | ports.in1
    xor = ports.in0 ^ ports.in1
    inv = ~ports.in0
    neg = -ports.in0


# -----


@unittestmodule()
class Multiple:
  in0 = Input(types.si16)
  in1 = Input(types.si16)

  @generator
  def construct(ports):
    ports.in0 + ports.in1 + ports.in0 + ports.in1


# -----


@unittestmodule(run_passes=False)
class Casts:
  in0 = Input(types.i16)

  @generator
  def construct(ports):
    in0s = ports.in0.asSInt
    in0u = ports.in0.asUInt
    in0s_i = in0s.asInt
