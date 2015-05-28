using System;
using System.IO;

namespace MNIST_Reader
{
    public class MNIST_Label_Reader
    {

        private const int HEADER_SIZE = 8;

        private byte dataType;
        private byte numOfDims;

        private int numOfLabels;
        private int labelsRead;

        private double[] nextLabel;

        private MemoryStream reader = null;

        public MNIST_Label_Reader(MemoryStream stream)
        {
            reader = stream;
            ReadHeader();
        }

        private bool CanRead()
        {
            return labelsRead < numOfLabels;
        }

        public double[] ReadNext()
        {
            if (CanRead())
            {
                if (nextLabel == null)
                    nextLabel = new double[10];

                labelsRead++;

                nextLabel[reader.ReadByte()] = 1;

                return nextLabel; 
            }
            return null;
        }

        private void ReadHeader()
        {
            byte[] header = new byte[HEADER_SIZE];

            reader.Read(header, 0, HEADER_SIZE);
            
            //Third byte is the data type
            dataType = header[2];
            //Fourth byte is the number of dimensions of input (1 if label, 3 if images)
            numOfDims = header[3];

            numOfLabels = ConvertToInt32(header, 4);
        }

        private int ConvertToInt32(byte[] source, int offset)
        {
            int value = 0;
            for (int i = 0; i < 4; i++)
                value |= source[offset + i] << (24 - 8 * i);

            return value;
        }    
    }
}
