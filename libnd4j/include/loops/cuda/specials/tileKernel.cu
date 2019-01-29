/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author GS <sgazeos@gmail.com>, created on 16.01.2019
//

#include <loops/special_kernels.h>

namespace nd4j {

////////////////////////////////////////////////////////////////////////
    template<typename T>
    static __global__ void
    tileKernel(void const *inputBuffer, Nd4jLong *inputShape, void *outputBuffer, Nd4jLong *outputShape,
               Nd4jLong resultLength) {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        Original code to transform in cuda-based
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;
        //const auto resultLength = shape::length(outputShape);
        if (shape::order(outputShape) == 'c') {           //  ews == 1 always here
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto yOffset = shape::subArrayIndex(outputShape, inputShape, i);
                *(reinterpret_cast<T *>(outputBuffer) + i) = *(reinterpret_cast<T const *>(inputBuffer) + yOffset);
            }
//            for(Nd4jLong i=0;  i<resultLen; ++i) {
//                auto yOffset = shape::subArrayIndex(newShapeInfo, _shapeInfo, i);
//                BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign, (newBuff, i, this->_buffer, yOffset), LIBND4J_TYPES);
//
//            }
        } else {
//
            //auto inputLength = shape::lenght(inputShape);
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto xOffset = shape::getIndexOffset(i, outputShape, resultLength);
                auto yOffset = shape::subArrayIndex(outputShape, inputShape, i);
                *(reinterpret_cast<T *>(outputBuffer) + xOffset) = *(reinterpret_cast<T const *>(inputBuffer) +
                                                                     yOffset);
//                BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign, (newBuff, xOffset, this->_buffer, yOffset), LIBND4J_TYPES);
            }
        }

    }

    BUILD_SINGLE_TEMPLATE(template __global__ void tileKernel,
                          (void const* inputBuffer, Nd4jLong* inputShape, void* outputBuffer, Nd4jLong* outputShape, Nd4jLong resultLength),
                          LIBND4J_TYPES);

    template<typename T>
    void tileKernelH(void const *inputBuffer, Nd4jLong *inputShape, void *outputBuffer, Nd4jLong *outputShape,
                     Nd4jLong resultLength, cudaStream_t stream) {
        dim3 launchDims(256, 512, 8192);
        tileKernel<T> << < launchDims.x, launchDims.y, launchDims.z, stream >> >
                                                                     (inputBuffer, inputShape, outputBuffer, outputShape, resultLength);
    }

    BUILD_SINGLE_TEMPLATE(template void tileKernelH,
                          (void const* inputBuffer, Nd4jLong* inputShape, void* outputBuffer, Nd4jLong* outputShape, Nd4jLong resultLength, cudaStream_t stream),
                          LIBND4J_TYPES);

    template<typename X, typename Y>
    static __global__ void
    tileKernelDouble(void const *inputBuffer, Nd4jLong *inputShape, void *outputBuffer, Nd4jLong *outputShape,
                     Nd4jLong resultLength, Nd4jLong ews) {
        char ordering = shape::order(outputShape);
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;

        if (ordering == 'c' && ews == 1) {           //  ews == 1 always here
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto yOffset = shape::subArrayIndex(outputShape, inputShape, i);
                *(reinterpret_cast<X *>(outputBuffer) + i) = static_cast<X>(*(reinterpret_cast<Y const *>(inputBuffer) +
                                                                              yOffset));
            }
        } else if (ordering == 'c' && ews > 1) {
            for (int i = tid; i < resultLength; i += totalThreads) {
                auto yOffset = shape::subArrayIndex(outputShape, inputShape, i);
                *(reinterpret_cast<X *>(outputBuffer) + i * ews) = static_cast<X>(*(
                        reinterpret_cast<Y const *>(inputBuffer) + yOffset));
            }
        } else {

            for (int i = tid; i < resultLength; i += totalThreads) {

                auto xOffset = shape::getIndexOffset(i, outputShape, resultLength);
                auto yOffset = shape::subArrayIndex(outputShape, inputShape, i);
                *(reinterpret_cast<X *>(outputBuffer) + xOffset) = static_cast<X>(*(
                        reinterpret_cast<Y const *>(inputBuffer) + yOffset));
            }
        }
    }

    BUILD_DOUBLE_TEMPLATE(template __global__ void tileKernelDouble,
                          (void const* inputBuffer, Nd4jLong* inputShape, void* outputBuffer, Nd4jLong* outputShape, Nd4jLong resultLength, Nd4jLong ews),
                          LIBND4J_TYPES, LIBND4J_TYPES);

    template<typename X, typename Y>
    void tileKernelHH(void const *inputBuffer, Nd4jLong *inputShape, void *outputBuffer, Nd4jLong *outputShape,
                      Nd4jLong resultLength, Nd4jLong ews, cudaStream_t stream) {
        dim3 launchDims(256, 512, 8192);
        tileKernelDouble<X, Y> << < launchDims.x, launchDims.y, launchDims.z, stream >> >
                                                                              (inputBuffer, inputShape, outputBuffer, outputShape, resultLength, ews);
    }

    BUILD_DOUBLE_TEMPLATE(template void tileKernelHH,
                          (void const* inputBuffer, Nd4jLong* inputShape, void* outputBuffer, Nd4jLong* outputShape, Nd4jLong resultLength, Nd4jLong ews, cudaStream_t stream),
                          LIBND4J_TYPES, LIBND4J_TYPES);



    template <typename Lambda>
    __global__ void runLambda(double *input, double *output, Nd4jLong length, Lambda lambda) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        float16 f(1.0f);
        for (Nd4jLong e = tid; e < length; e += gridDim.x * blockDim.x) {
            output[e] = lambda(input[e]) + (double) f;
        }
    }

    void launcher(cudaStream_t *stream, double *input, double *output, Nd4jLong length) {
        auto f = [] __device__ (double x) -> double {
            return x + 1.;
        };

        runLambda<<<128, 128, 128, *stream>>>(input, output, length, f);
    }
}