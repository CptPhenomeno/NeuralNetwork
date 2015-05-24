using System;
using System.IO;
using System.Collections.Generic;


namespace DatasetUtility
{
    class Test
    {

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

        static Dataset ReadMonkDataset(StringReader stream, int numOfInput, int numOfOutput, int numOfFold)
        {
            try
            {
                Dataset monk = new Dataset(numOfFold, 124 / numOfFold, numOfInput, numOfOutput);
                string trainingExample = stream.ReadLine();
                trainingExample.TrimEnd();
                char[] separator = { ' ' };

                while (trainingExample != null)
                {
                    string[] strings = trainingExample.Split(separator);
                    double[] output = { Double.Parse(strings[0]) };
                    List<string> stringList = new List<string>(strings);
                    strings = stringList.GetRange(1, strings.Length - 1).ToArray();
                    double[] input = EncodingInput(strings);

                    monk.AddSampleToDataset(new Sample(input, output));

                    trainingExample = stream.ReadLine();
                }
                return monk;
            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        static void Main()
        {
            using (StringReader str = new StringReader(Properties.Resources.monks_1_train))
            {

                Dataset monk = ReadMonkDataset(str, 17, 1, 4);
                Fold val = monk.ValidationFold;
                Console.WriteLine();
            }
        }
    }
}
