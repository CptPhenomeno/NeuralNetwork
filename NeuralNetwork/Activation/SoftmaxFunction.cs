namespace NeuralNetwork.Activation
{
    using System;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class SoftmaxFunction : IActivationFunction
    {
        public SoftmaxFunction() : base() { }

        public Vector<double> Function(Vector<double> x)
        {
            Vector<double> expVector = x.Map(e => Math.Exp(e));
            double sum = expVector.Sum();
            return expVector.Map(e => e / sum);
        }

        public Vector<double> Derivative(Vector<double> x)
        {
            Vector<double> y = Function(x);
            return (y.Map(e => e * (1.0 - e)));
        }
    }
}
