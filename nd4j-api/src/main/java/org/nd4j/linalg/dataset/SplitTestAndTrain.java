/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.dataset;

/**
 * Split test and train
 *
 * @author Adam Gibson
 */
public class SplitTestAndTrain {

    private DataSet train, test;

    public SplitTestAndTrain(DataSet train, DataSet test) {

        this.train = train;
        this.test = test;
    }

    public DataSet getTest() {
        return test;
    }

    public void setTest(DataSet test) {
        this.test = test;
    }

    public DataSet getTrain() {

        return train;
    }

    public void setTrain(DataSet train) {
        this.train = train;
    }
}
