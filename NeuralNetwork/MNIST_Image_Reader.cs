using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace MNIST_Reader
{
    public class MNIST_Image_Reader
    {
        private const int HEADER_SIZE = 16;

        private byte dataType;
        private byte numOfDims;
        private int numOfImages;
        private int imagesRead;
        private int rows;
        private int columns;

        private double[] nextPixels = null;

        private MemoryStream reader = null;

        public MNIST_Image_Reader(MemoryStream stream)
        {
            reader = stream;
            ReadHeader();
        }

        private bool CanRead()
        {
            return imagesRead < numOfImages;
        }

        public double[] ReadNext()
        {
            if (CanRead())
            {
                if (nextPixels == null)
                    nextPixels = new double[rows * columns];
                else
                    Array.Clear(nextPixels, 0, nextPixels.Length);

                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < columns; c++)
                    {
                        nextPixels[columns*r + c] = (double)(reader.ReadByte() & 0xFF);
                    }
                }

                imagesRead++;

                return nextPixels;
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

            numOfImages = ConvertToInt32(header, 4);
            rows = ConvertToInt32(header, 8);
            columns = ConvertToInt32(header, 12);
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
