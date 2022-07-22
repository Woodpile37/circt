

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "assert.h"

using namespace mlir;
using namespace circt;

namespace circt {

namespace cde {

// A context containing the shared state of a generator.
struct CDEContext {
  OpBuilder &b;
  Value clk;
  Value rst;
};

// Base class for CDE values. Users can specialize this class for
// dialect-specific operations.
class CDEValue {

  template <typename TBinOp>
  CDEValue execBinOp(CDEValue &other) {
    return CDEValue(ctx, ctx.b.create<TBinOp>(v.getLoc(), self, other));
  }

public:
  CDEValue(CDEContext &ctx, Value v) : ctx(ctx), v(v) {}
  virtual ~CDEValue() {}

  using TAnd = comb::AndOp;
  using TOr = comb::OrOp;
  using TXor = comb::XorOp;

  Value operator Value() { return v; }

  virtual CDEValue operator+(CDEValue other) {
    return execBinOp<comb::AddOp>(other);
  }
  virtual CDEValue operator-(CDEValue other) {
    return execBinOp<comb::SubOp>(other);
  }
  virtual CDEValue operator*(CDEValue other) {
    return execBinOp<comb::MulOp>(other);
  }
  virtual CDEValue operator<<(CDEValue other) {
    return execBinOp<comb::ShlOp>(other);
  }
  virtual CDEValue operator&(CDEValue other) {
    return execBinOp<comb::AndOp>(other);
  }
  virtual CDEValue operator|(CDEValue other) {
    return execBinOp<comb::OrOp>(other);
  }
  virtual CDEValue operator^(CDEValue other) {
    return execBinOp<comb::XorOp>(other);
  }
  virtual CDEValue operator~() {
    return CDEValue(ctx, comb::createOrFoldNot(v.getLoc(), v, ctx.b));
  }
  virtual CDEValue shrs(CDEValue other) {
    return execBinOp<comb::ShrSOp>(other);
  }
  virtual CDEValue shru(CDEValue other) {
    return execBinOp<comb::ShrUOp>(other);
  }
  virtual CDEValue divu(CDEValue other) {
    return execBinOp<comb::DivUOp>(other);
  }
  virtual CDEValue divs(CDEValue other) {
    return execBinOp<comb::DivSOp>(other);
  }
  virtual CDEValue modu(CDEValue other) {
    return execBinOp<comb::ModUOp>(other);
  }
  virtual CDEValue mods(CDEValue other) {
    return execBinOp<comb::ModSOp>(other);
  }
  virtual CDEValue concat(CDEValue other) {
    return execBinOp<comb::ConcatOp>(other);
  }

  virtual CDEValue reg(StringRef name) {
    assert(ctx.clk && "A clock must be in the CDE context to creae a "
                      "register.");
    return CDEValue(ctx,
                    ctx.b.create<seq::CompRegOp>(v.getLoc(), v, ctx.clk,
                                                 ctx.b.getStringAttr(name)));
  }

protected:
  CDEContext &ctx;
  Value v;
};

// A class for providing backedge-based access to the in- and output ports of
// a module.
class CDEPorts {
  // ...
};

// A CDE Generator is a utility class for supporting DSL-like building of CIRCT
// RTL dialect operations.
template <typename TValue = CDEValue>
class Generator {

private:
  /// Unwraps a range of CDEValues to their underlying values.
  static llvm::SmallVector<Value> unwrap(llvm::ArrayRef<TValue> values) {
    SmallVector<Value, 4> unwrapped;
    llvm::transform(operands, std::back_inserter(unwrapped),
                    [](TValue v) { return v.v; });
    return unwrapped;
  }

  template <typename TOp>
  CDEValue execNAryOp(llvm::ArrayRef<TValue> operands) {
    SmallVector<Value> unwrapped = unwrap(operands);
    return TValue(ctx, ctx.b.create<TOp>(v.getLoc(), unwrapped));
  }

  // Implementation-defined generator function.
  virtual void generate(CDEPorts &ports) = 0;

public:
  Generator(OpBuilder &b) {
    ctx.b = b;
    ctx.clk = nullptr;
    ctx.rst = nullptr;
  }

  // N-ary operations.
  TValue And(llvm::ArrayRef<TValue> operands) {
    return execNAryOp<TValue::TAnd>(operands);
  }

  TValue Or(llvm::ArrayRef<TValue> operands) {
    return execNAryOp<TValue::TOr>(operands);
  }

  TValue Xor(llvm::ArrayRef<TValue> operands) {
    return execNAryOp<TValue::TXor>(operands);
  }

private:
  CDEContext ctx;
};

/// A CDE Generator which generates entire modules.
template <typename TValue>
class CDEModuleGenerator : public cde::Generator<TValue> {

public:
protected:
  void init() {
    auto &[ins, outs] = getIO();

    // Create the module.
    auto module = ctx.b.create<ModuleOp>(v.getLoc());
  }
  virtual std::pair<TypeRange, TypeRange> getIO() = 0;

  hw::HWModuleOp mod;

}; // namespace cde

class ESIValue : public cde::CDEValue {
public:
  using cde::CDEValue::CDEValue;

  std::pair<ESIValue, ESIValue> wrap(ESIValue &rawInput, ESIValue &valid) {
    auto wrapOp =
        ctx.b.create<esi::WrapValidReady>(v.getLoc(), rawInput.v, valid.v);
    return std::make_pair(ESIValue(ctx, wrapOp.getResult(0)),
                          ESIValue(ctx, wrapOp.getResult(1)));
  }

  std::pair<ESIValue, ESIValue> unwrap(ESIValue &chanInput, ESIValue &ready) {
    auto unwrapOp =
        ctx.b.create<esi::UnwrapValidReady>(v.getLoc(), chanInput.v, ready.v);
    return std::make_pair(ESIValue(ctx, unwrapOp.getResult(0)),
                          ESIValue(ctx, unwrapOp.getResult(1)));
  }
};

class HandshakeGenerator : public cde::ModuleGenerator<ESIValue> {
  // A base class for handshake generator functions.
  HandshakeGenerator(OpBuilder &b) {}

  std::pair<TypeRange, TypeRange> getIO() override {
    // Create ESI ports for the requested in- and outputs.
    // ...
    return {};
  }
};

class ForkGenerator : public HandshakeGenerator {
public:
  ForkGenerator(OpBuilder &b, Type inner, size_t nOutputs)
      : HandshakeGenerator(b, nInputs, nOutputs) {
    llvm::SmallVector<Type> ins, outs;
    // Inputs: A single ESI channel + clk, reset
    auto channelType = esi::PortType::get(inner);
    ins.push_back(channelType);
    ins.append({b.getI1Type(), b.getI1Type()});
    // Ouputs: nOutput ESI channels
    for (int i = 0; i < nOutputs; i++)
      outs.push_back(channelType);
    // some initialization...
  }

  // Generator function, functions much like PyCDE generators, but ports are
  // accessed through a string-based lookup. Probably the best (only?) way to
  // access the dynamically created ports for the underlying module.
  void generate(CDEPorts &ports) override {
    BackedgeBuilder bb(ctx.b); // possibly a CDE backedge builder which
                               // automatically wraps to CDEValue's.
    auto allDoneBE = bb.get(types.i1);
    auto &[src_data, src_valid] = ports["in0"].unwrap(allDoneBE);
    llvm::SmallVector<Value> done_wires;

    for (auto &[i, outport] : llvm::enumerate(ports.outputs)) {
      auto doneBE = bb.get(types.i1);
      auto outReadyBE = bb.get(types.i1);

      // The value has been emitted when {doneWire && notallDoneWire}. Only if
      // notallDone, the emtdReg will be set to the value of doneWire.
      // Otherwise, all emtdRegs will be cleared to zero.
      auto emitted = doneBE & ~allDoneBE;

      // Create an emitted register.
      auto emitted_reg = emitted.reg("emitted");

      // Create valid signal and connect to the result valid. The reason of this
      // 'and'' is each result can only be emitted once.
      auto out_valid = ~emitted & src_valid;

      // Create validReady wire signal, which indicates a successful handshake
      // in the current clock cycle.
      auto valid_ready = outReadyBE & out_valid;
      auto done = valid_ready | emitted_reg;
      doneBE.setValue(done);
      done_wires.push_back(done);

      // Wrap the output esi port and set the output ready backedge.
      auto out_ready = outport.wrap(src_data, out_valid);
      outReadyBE.setValue(out_ready);
    }

    allDoneBE.setValue(And(done_wires));
  }
};

} // namespace cde

} // namespace circt