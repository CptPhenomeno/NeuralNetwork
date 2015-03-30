namespace NeuralNetwork.Learning
{
    using NeuralNetwork.Network;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;
    
    using System;
    using System.Collections.Concurrent;
    
    public class BackPropagationTrainer
    {
        private BackPropagation backpropagation;
        private BlockingCollection<string> errorOnEpochs;
        private bool logInformationEnabled;
        
        private double maxError;
        private int maxEpoch;
        private int batchSize;

        private bool running;

        public BackPropagationTrainer(NeuralNet net, double learningRate = 0.3, double momentum = 0.0, double maxError = 0.01,
                                      int maxEpoch = 1000, int batchSize = 1)
        {
            backpropagation = new BackPropagation(net, learningRate, momentum, batchSize);
            logInformationEnabled = false;
            running = false;
            this.maxError = maxError;
            this.maxEpoch = maxEpoch;
            this.batchSize = batchSize;
        }

        public void Learn(double[][] inputs, double[][] outputs, double[][] testSet = null)
        {
            Vector<double>[] vectorInputs = new Vector<double>[inputs.Length];
            Vector<double>[] vectorOutputs = new Vector<double>[outputs.Length];
            Vector<double>[] vectorTest = null;

            int inputIndex = 0, outputIndex = 0;

            foreach (double[] input in inputs)
                vectorInputs[inputIndex++] = Vector<double>.Build.DenseOfArray(input);

            foreach (double[] output in outputs)
                vectorOutputs[outputIndex++] = Vector<double>.Build.DenseOfArray(output);

            if (testSet != null)
            {
                int testIndex = 0;
                foreach (double[] test in testSet)
                    vectorOutputs[testIndex++] = Vector<double>.Build.DenseOfArray(test);
            }

            Learn(vectorInputs, vectorOutputs, vectorTest);
        }

        public void Learn(Vector<double>[] inputs, Vector<double>[] expectedOutputs, Vector<double>[] testSet = null)
        {
            running = true;
            double error = 0.0;
            int epoch = -1;
            int numOfElemInEpoch = inputs.Length;

            do
            {
                ++epoch;
                error = RunEpoch(inputs, expectedOutputs);
                error /= numOfElemInEpoch;
                if (LogInformationEnabled)
                    errorOnEpochs.Add(epoch + ":" + error);
            } while (error > MaxError && epoch < MaxEpoch);
            if (logInformationEnabled)
                errorOnEpochs.CompleteAdding();
            running = false;
        }

        private double RunEpoch(Vector<double>[] inputs, Vector<double>[] expectedOutputs)
        {
            double error = 0.0;

            for (int next = 0; next < inputs.Length; next += batchSize)
            {
                error += backpropagation.Run(next, inputs, expectedOutputs);
                //Update the network only after batchSize examples
                backpropagation.UpdateNetwork();
            }

            return error;
        }

        #region Getter & Setter

        public double MaxError
        {
            get { return maxError; }
            set { maxError = value; }
        }

        public int MaxEpoch
        {
            get { return maxEpoch; }
            set { maxEpoch = value; }
        }

        public int BatchSize
        {
            get { return batchSize; }
            set { batchSize = value; }
        }

        public bool LogInformationEnabled
        {
            get { return logInformationEnabled; }
            set 
            {
                if (!running)
                    logInformationEnabled = value;
                else
                    Console.WriteLine("Cannot enable/disable log when the learning run!");
            }
        }

        public void EnableLogging(BlockingCollection<string> logStorage)
        {
            LogInformationEnabled = true;
            errorOnEpochs = logStorage;
        }

        public void DisableLogging()
        {
            LogInformationEnabled = false;
        }

        #endregion

        #region BackPropagation Getter & Setter

        public double LearningRate
        {
            get { return backpropagation.LearningRate; }
            set { backpropagation.LearningRate = value; }
        }

        public double Momentum
        {
            get { return backpropagation.Momentum; }
            set { backpropagation.Momentum = value; }
        }

        #endregion
    }
}
