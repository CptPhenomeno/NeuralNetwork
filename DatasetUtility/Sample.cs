using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace DatasetUtility
{
    public class Sample
    {
        private Vector<double> Input { get; set; }
        private Vector<double> Output { get; set; }

        public Sample(Vector<double> input, Vector<double> output)
        {
            Input = input;
            Output = output;
        }

        public Sample(double[] input, double[] output)
        {
            Vector<double> inputVector = Vector<double>.Build.DenseOfArray(input);
            Vector<double> outputVector = Vector<double>.Build.DenseOfArray(output);

            Input = inputVector;
            Output = outputVector;
        }
    }
}
