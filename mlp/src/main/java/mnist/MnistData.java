package mnist;

import java.io.*;

public class MnistData {
    private final int numberOfItems;
    private final int rows;
    private final int cols;
    private final double[][] input;
    private final double[][] output;

    public MnistData(final String imageFileName, final String labelFileName) {
        try ( DataInputStream images = new DataInputStream(new FileInputStream(imageFileName));
              DataInputStream labels = new DataInputStream(new FileInputStream(labelFileName)) ) {
            if (images.readInt() != 2051) {
                throw new IllegalArgumentException("Invalid image file: " + imageFileName);
            }
            if (labels.readInt() != 2049) {
                throw new IllegalArgumentException("Invalid labels file" + labelFileName);
            }
            numberOfItems = labels.readInt();
            if (images.readInt() != numberOfItems) {
                throw new IllegalArgumentException("Number of items does not match");
            }

            // initialize input and output
            input = new double[numberOfItems][];
            output = new double[numberOfItems][];

            // read labels
            byte[] labelData = new byte[numberOfItems];
            labels.read(labelData);
            for (int j = 0; j < numberOfItems; j++) {
                output[j] = new double[10];
                for (int i = 0; i < 10; i++) {
                    if (labelData[j] == i) {
                        output[j][i] = 1.0;
                    }
                    else {
                        output[j][i] = 0.0;
                    }
                }
            }

            // read images
            rows = images.readInt();
            cols = images.readInt();
            int imageSize = rows * cols;
            for (int i = 0; i < numberOfItems; i++) {
                byte[] imageData = new byte[imageSize];
                input[i] = new double[imageSize];
                images.read(imageData);
                for (int j = 0; j < imageSize; j++) {
                    int x = imageData[j] & 0xff;
                    input[i][j] = x / 255.0;
                }
            }
        }
        catch (final IOException ex) {
            throw new IllegalArgumentException("Can't read file", ex);
        }
    }

    public int numberOfItems() {
        return numberOfItems;
    }

    public double[] input(final int i) {
        return input[i];
    }

    public double[] output(final int i) {
        return output[i];
    }

    public void display(final int index) {
        int x = 0;
        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < cols; i++) {
                double d = input(index)[x++];
                if      (d < 0.2) System.out.print(" ");
                else if (d < 0.5) System.out.print(".");
                else if (d < 0.8) System.out.print("-");
                else              System.out.print("=");
            }
            System.out.println();
        }
        System.out.println("----------------------------");
    }
}
