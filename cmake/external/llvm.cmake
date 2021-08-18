# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

# released on 04/15/2021
# TODO(zhangting2020): download the corresponding package according to the environment
set(LLVM_VER   "12.0.0")
set(LLVM_URL "${GIT_URL}/llvm/llvm-project/releases/download/llvmorg-${LLVM_VER}/clang+llvm-${LLVM_VER}-x86_64-linux-gnu-ubuntu-16.04.tar.xz" CACHE STRING "" FORCE)

MESSAGE(STATUS "LLVM_VERSION: ${LLVM_VER}, LLVM_URL: ${LLVM_URL}")

set(LLVM_PREFIX_DIR ${THIRD_PARTY_PATH}/llvm)
set(LLVM_SOURCE_DIR ${THIRD_PARTY_PATH}/llvm/src/extern_llvm)
cache_third_party(extern_llvm
        URL       ${LLVM_URL}
        DIR       LLVM_SOURCE_DIR)

ExternalProject_Add(
    extern_llvm
    ${EXTERNAL_PROJECT_LOG_ARGS}
    "${LLVM_DOWNLOAD_CMD}"
    URL_MD5               20c20a6fa716e7f376c954e957b0b218
    PREFIX                ${LLVM_PREFIX_DIR}
    DOWNLOAD_DIR          ${LLVM_SOURCE_DIR}
    SOURCE_DIR            ${LLVM_SOURCE_DIR}
    DOWNLOAD_NO_PROGRESS  1
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    INSTALL_COMMAND       ""
    UPDATE_COMMAND        ""
    )

set(LLVM_INCLUDE_DIRS ${LLVM_SOURCE_DIR}/include)
set(LLVM_LIBRARY_DIRS ${LLVM_SOURCE_DIR}/lib)
# "llvm-config --cxxflags" get the LLVM_DEFINITIONS
# "llvm-config --libs" get the LLVM_LIBS. We temporarily use a fixed string
# because llvm has not been downloaded yet in the this stage
set(LLVM_DEFINITIONS "-D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS")
string(CONCAT LLVM_LIBS "-lLLVMWindowsManifest -lLLVMXRay -lLLVMLibDriver "
                        "-lLLVMDlltoolDriver -lLLVMCoverage -lLLVMLineEditor "
                        "-lLLVMXCoreDisassembler -lLLVMXCoreCodeGen -lLLVMXCoreDesc "
                        "-lLLVMXCoreInfo -lLLVMX86Disassembler -lLLVMX86AsmParser "
                        "-lLLVMX86CodeGen -lLLVMX86Desc -lLLVMX86Info "
                        "-lLLVMWebAssemblyDisassembler -lLLVMWebAssemblyAsmParser "
                        "-lLLVMWebAssemblyCodeGen -lLLVMWebAssemblyDesc -lLLVMWebAssemblyInfo "
                        "-lLLVMSystemZDisassembler -lLLVMSystemZAsmParser -lLLVMSystemZCodeGen "
                        "-lLLVMSystemZDesc -lLLVMSystemZInfo -lLLVMSparcDisassembler "
                        "-lLLVMSparcAsmParser -lLLVMSparcCodeGen -lLLVMSparcDesc "
                        "-lLLVMSparcInfo -lLLVMRISCVDisassembler -lLLVMRISCVAsmParser "
                        "-lLLVMRISCVCodeGen -lLLVMRISCVDesc -lLLVMRISCVInfo "
                        "-lLLVMPowerPCDisassembler -lLLVMPowerPCAsmParser -lLLVMPowerPCCodeGen "
                        "-lLLVMPowerPCDesc -lLLVMPowerPCInfo -lLLVMNVPTXCodeGen -lLLVMNVPTXDesc "
                        "-lLLVMNVPTXInfo -lLLVMMSP430Disassembler -lLLVMMSP430AsmParser "
                        "-lLLVMMSP430CodeGen -lLLVMMSP430Desc -lLLVMMSP430Info "
                        "-lLLVMMipsDisassembler -lLLVMMipsAsmParser -lLLVMMipsCodeGen "
                        "-lLLVMMipsDesc -lLLVMMipsInfo -lLLVMLanaiDisassembler -lLLVMLanaiCodeGen "
                        "-lLLVMLanaiAsmParser -lLLVMLanaiDesc -lLLVMLanaiInfo "
                        "-lLLVMHexagonDisassembler -lLLVMHexagonCodeGen -lLLVMHexagonAsmParser "
                        "-lLLVMHexagonDesc -lLLVMHexagonInfo -lLLVMBPFDisassembler "
                        "-lLLVMBPFAsmParser -lLLVMBPFCodeGen -lLLVMBPFDesc -lLLVMBPFInfo "
                        "-lLLVMAVRDisassembler -lLLVMAVRAsmParser -lLLVMAVRCodeGen -lLLVMAVRDesc "
                        "-lLLVMAVRInfo -lLLVMARMDisassembler -lLLVMARMAsmParser -lLLVMARMCodeGen "
                        "-lLLVMARMDesc -lLLVMARMUtils -lLLVMARMInfo -lLLVMAMDGPUDisassembler "
                        "-lLLVMAMDGPUAsmParser -lLLVMAMDGPUCodeGen -lLLVMAMDGPUDesc "
                        "-lLLVMAMDGPUUtils -lLLVMAMDGPUInfo -lLLVMAArch64Disassembler "
                        "-lLLVMAArch64AsmParser -lLLVMAArch64CodeGen -lLLVMAArch64Desc "
                        "-lLLVMAArch64Utils -lLLVMAArch64Info -lLLVMOrcJIT -lLLVMMCJIT "
                        "-lLLVMJITLink -lLLVMOrcTargetProcess -lLLVMOrcShared -lLLVMInterpreter "
                        "-lLLVMExecutionEngine -lLLVMRuntimeDyld -lLLVMSymbolize "
                        "-lLLVMDebugInfoPDB -lLLVMDebugInfoGSYM -lLLVMOption -lLLVMObjectYAML "
                        "-lLLVMMCA -lLLVMMCDisassembler -lLLVMLTO -lLLVMCFGuard "
                        "-lLLVMFrontendOpenACC -lLLVMExtensions -lPolly -lPollyISL "
                        "-lLLVMPasses -lLLVMObjCARCOpts -lLLVMHelloNew -lLLVMCoroutines "
                        "-lLLVMipo -lLLVMInstrumentation -lLLVMVectorize -lLLVMLinker "
                        "-lLLVMFrontendOpenMP -lLLVMDWARFLinker -lLLVMGlobalISel -lLLVMMIRParser "
                        "-lLLVMAsmPrinter -lLLVMDebugInfoDWARF -lLLVMSelectionDAG -lLLVMCodeGen "
                        "-lLLVMIRReader -lLLVMAsmParser -lLLVMInterfaceStub -lLLVMFileCheck "
                        "-lLLVMFuzzMutate -lLLVMTarget -lLLVMScalarOpts -lLLVMInstCombine "
                        "-lLLVMAggressiveInstCombine -lLLVMTransformUtils -lLLVMBitWriter "
                        "-lLLVMAnalysis -lLLVMProfileData -lLLVMObject -lLLVMTextAPI "
                        "-lLLVMMCParser -lLLVMMC -lLLVMDebugInfoCodeView -lLLVMDebugInfoMSF "
                        "-lLLVMBitReader -lLLVMCore -lLLVMRemarks -lLLVMBitstreamReader "
                        "-lLLVMBinaryFormat -lLLVMTableGen -lLLVMSupport -lLLVMDemangle "
                        "-lrt -ldl -lpthread -lm -lz -ltinfo")
include_directories(BEFORE SYSTEM ${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_library(llvm INTERFACE)
add_dependencies(llvm extern_llvm)
