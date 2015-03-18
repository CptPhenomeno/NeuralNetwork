using System;
using System.IO;
using System.Reflection;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using NeuralNetwork.Network;
using NeuralNetwork.Learning;
using NeuralNetwork.Activation;

namespace NeuralNetwork
{
    static class Program
    {

        private static void garbageTest()
        {
            int[] layerSize = { 3, 1 };
            IActivationFunction[] functions = { new SigmoidFunction(), new Threshold(0.5) };

            NeuralNet net = new NeuralNet(2, layerSize, functions);

            Matrix<double> input1 = Matrix<double>.Build.DenseOfArray(new[,] { { 0.0 }, { 0.0 } });
            Matrix<double> input2 = Matrix<double>.Build.DenseOfArray(new[,] { { 0.0 }, { 1.0 } });
            Matrix<double> input3 = Matrix<double>.Build.DenseOfArray(new[,] { { 1.0 }, { 0.0 } });
            Matrix<double> input4 = Matrix<double>.Build.DenseOfArray(new[,] { { 1.0 }, { 1.0 } });

            net.ComputeOutput(input1);
            Console.WriteLine("output1 {0}", net.Output);

            net.ComputeOutput(input2);
            Console.WriteLine("output2 {0}", net.Output);

            net.ComputeOutput(input3);
            Console.WriteLine("output3 {0}", net.Output);

            net.ComputeOutput(input4);
            Console.WriteLine("output4 {0}", net.Output);

            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new[,] { { 1d, 2d, 3d }, { 4d, 5d, 6d } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new[,] { { 2d }, { 3d }, { 4d } });

            Console.WriteLine(m1);
            Console.WriteLine(m2);
            int i = 0;
            m1.MapInplace((x) =>
            {
                return x * m2.At((i++) / 2, 0);
            });
            Console.WriteLine(m1);

            Matrix<double> test = Matrix<double>.Build.DenseOfArray(new[,] { { 1d, 2d, 3d, 4d }, { 5d, 6d, 7d, 8d } });
            Matrix<double> mul = Matrix<double>.Build.DenseOfArray(new[,] { { 10d }, { 20d } });
            Console.WriteLine(test);
            Console.WriteLine(mul);
            Console.WriteLine(test.Transpose().Multiply(mul));

            Matrix<double> m = Matrix<double>.Build.Random(3, 5);
            int indexMax = 0;
            double max = -1000;
            for (int r = 0; r < m.RowCount; r++)
                for (int c = 0; c < m.ColumnCount; c++)
                {
                    double el = m.At(r, c);
                    if (el > max)
                    {
                        max = el;
                        indexMax = r * m.ColumnCount + c;
                    }
                }
            Console.WriteLine(m);
            Console.WriteLine("max: {0} in position {1}, row {2} column {3}", max, indexMax,
                (indexMax / m.ColumnCount), (indexMax % m.ColumnCount));
        }

        private static void TestXOR()
        {
            int[] layerSize = { 3, 1 };
            IActivationFunction[] functions = { new SigmoidFunction(), new HyperbolicTangentFunction() };

            NeuralNet net = new NeuralNet(2, layerSize, functions);

            #region Input
            Matrix<double> input1 = Matrix<double>.Build.DenseOfArray(new[,] { { 0.0 }, { 0.0 } });
            Matrix<double> input2 = Matrix<double>.Build.DenseOfArray(new[,] { { 0.0 }, { 1.0 } });
            Matrix<double> input3 = Matrix<double>.Build.DenseOfArray(new[,] { { 1.0 }, { 0.0 } });
            Matrix<double> input4 = Matrix<double>.Build.DenseOfArray(new[,] { { 1.0 }, { 1.0 } });

            Matrix<double>[] inputs = { input1, input2, input3, input4 };
            #endregion

            #region Output
            Matrix<double> output1 = Matrix<double>.Build.DenseOfArray(new[,] { { 0.0 } });
            Matrix<double> output2 = Matrix<double>.Build.DenseOfArray(new[,] { { 1.0 } });
            Matrix<double> output3 = Matrix<double>.Build.DenseOfArray(new[,] { { 1.0 } });
            Matrix<double> output4 = Matrix<double>.Build.DenseOfArray(new[,] { { 0.0 } });

            Matrix<double>[] outputs = { output1, output2, output3, output4 };
            #endregion

            Console.WriteLine(  "****************\n"+
                                "*    Before    *\n"+
                                "****************\n");
            net.ComputeOutput(input1);
            Console.WriteLine("output1 {0} - atteso {1}", net.Output.At(0, 0), 0);

            net.ComputeOutput(input2);
            Console.WriteLine("output2 {0} - atteso {1}", net.Output.At(0, 0), 1);

            net.ComputeOutput(input3);
            Console.WriteLine("output3 {0} - atteso {1}", net.Output.At(0, 0), 1);

            net.ComputeOutput(input4);
            Console.WriteLine("output4 {0} - atteso {1}", net.Output.At(0, 0), 0);

            BackPropagation bprop = new BackPropagation(net);
            bprop.LearningRate = 0.7;
            bprop.learn(inputs, outputs);

            Console.WriteLine(  "\n****************\n"+
                                "*     After    *\n"+
                                "****************\n");
            net.ComputeOutput(input1);
            Console.WriteLine("output1 {0} - atteso {1}", net.Output.At(0,0), 0);
                                                                    
            net.ComputeOutput(input2);
            Console.WriteLine("output2 {0} - atteso {1}", net.Output.At(0, 0), 1);
                                                                      
            net.ComputeOutput(input3);
            Console.WriteLine("output3 {0} - atteso {1}", net.Output.At(0, 0), 1);
                                                                    
            net.ComputeOutput(input4);
            Console.WriteLine("output4 {0} - atteso {1}", net.Output.At(0, 0), 0);
        }

        private static void TestMonk()
        {
            Tuple<double[][], double[][]> dataset;
            Tuple<double[][], double[][]> testset;

            double[][] trainingExamples;
            double[][] expectedOutputs;

            double[][] testInput;
            double[][] testOutput;

            #region Testing Monk Dataset 1
            int[] layerSize = { 3, 1 };
            IActivationFunction[] functions = { new SigmoidFunction(), new SigmoidFunction() };
            NeuralNet net = new NeuralNet(17, layerSize, functions);
            BackPropagation backProp = new BackPropagation(net, 0.2,0.6);            

            
            using (StringReader trainSet = new StringReader(Properties.Resources.monks_1_train))
            using (StringReader testSet = new StringReader(Properties.Resources.monks_1_test))
            {
                dataset = ReadDataset(trainSet);
                testset = ReadDataset(testSet);

                trainingExamples = dataset.Item1;
                expectedOutputs = dataset.Item2;

                testInput = testset.Item1;
                testOutput = testset.Item2;             

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 1");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.Write("Train the network...");
                backProp.learn(ToMatrix(trainingExamples), 
                                    ToMatrix(expectedOutputs));
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.WriteLine("*******************");
            }
            #endregion

            #region Testing Monk Dataset 2
            layerSize = new[] { 3, 1 };            
            net = new NeuralNet(17, layerSize, functions);
            backProp = new BackPropagation(net, 0.1, 0.5);


            using (StringReader trainSet = new StringReader(Properties.Resources.monks_2_train))
            using (StringReader testSet = new StringReader(Properties.Resources.monks_2_test))
            {
                dataset = ReadDataset(trainSet);
                testset = ReadDataset(testSet);

                trainingExamples = dataset.Item1;
                expectedOutputs = dataset.Item2;

                testInput = testset.Item1;
                testOutput = testset.Item2;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 2");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.Write("Train the network...");
                backProp.learn(ToMatrix(trainingExamples),
                                    ToMatrix(expectedOutputs));
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.WriteLine("*******************");
            }
            #endregion

            #region Testing Monk Dataset 3
            layerSize = new[] { 3, 1 };
            net = new NeuralNet(17, layerSize, functions);
            backProp = new BackPropagation(net, 0.7, 0.3);


            using (StringReader trainSet = new StringReader(Properties.Resources.monks_3_train))
            using (StringReader testSet = new StringReader(Properties.Resources.monks_3_test))
            {
                dataset = ReadDataset(trainSet);
                testset = ReadDataset(testSet);

                trainingExamples = dataset.Item1;
                expectedOutputs = dataset.Item2;

                testInput = testset.Item1;
                testOutput = testset.Item2;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 3");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.Write("Train the network...");
                backProp.learn(ToMatrix(trainingExamples),
                                    ToMatrix(expectedOutputs));
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.WriteLine("*******************");
            }
            #endregion
        }

        private static Matrix<double>[] ToMatrix(double[][] array)
        {
            Matrix<double>[] m = new Matrix<double>[array.Length];
            for (int i = 0; i < array.Length; i++)
                m[i] = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(array[i]));
            return m;
        }

        static double[] Encode(int value, byte encode)
        {
            double[] encoded = new double[encode];
            encoded[value - 1] = 1;
            return encoded;
        }

        static double[] EncodingInput(string[] inputs)
        {
            List<double> input = new List<double>();
            byte[] encoding = { 3, 3, 2, 3, 4, 2 };

            for (int i = 0; i < inputs.Length; i++)
            {
                input.InsertRange(input.Count, Encode(Int32.Parse(inputs[i]), encoding[i]));
            }
            return input.ToArray();
        }

        static Tuple<double[][], double[][]> ReadDataset(StringReader stream)
        {
            try
            {
                string trainingExample = stream.ReadLine();
                trainingExample.TrimEnd();
                char[] separator = { ' ' };

                List<double[]> inputs = new List<double[]>();
                List<double[]> outputs = new List<double[]>();
                int c = 0;
                while (trainingExample != null)
                {
                    ++c;
                    string[] strings = trainingExample.Split(separator);
                    double[] output = { Double.Parse(strings[0]) };
                    List<string> stringList = new List<string>(strings);
                    strings = stringList.GetRange(1, strings.Length - 1).ToArray();
                    double[] input = EncodingInput(strings);

                    inputs.Add(input);
                    outputs.Add(output);

                    trainingExample = stream.ReadLine();
                }

                double[][] inputList = inputs.ToArray();
                double[][] outputList = outputs.ToArray();

                return new Tuple<double[][], double[][]>(inputList, outputList);


            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        private static double RunMonkTest(NeuralNet net, double[][] input, double[][] output)
        {
            double successRatio = 0.0;
            int numOfExamples = input.Length;

            for (int i = 0; i < input.Length; i++)
            {
                net.ComputeOutput(Matrix<double>.Build.DenseOfColumnArrays(input[i]));
                double netOutput = net.Output.At(0,0);
                double roundOutput = Math.Round(netOutput);
                double expected = output[i][0];

                successRatio += (expected == roundOutput) ? 1 : 0;
            }

            return successRatio / (double)numOfExamples * 100.0;
        }

        static void Main()
        {
            //TestXOR();
            TestMonk();
        }

    }
}
