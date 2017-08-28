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
            for (int i = 0; i < numberOfItems; i++) {
                output[i] = new double[1];
                output[i][0] = labelData[i];
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
                    input[i][j] = imageData[j] == 0 ? 0.0 : 1.0;
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
        int label = (int)(output[index][0]);
        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < cols; i++) {
                System.out.print(input(index)[x++] < 0.5 ? " ": label);
            }
            System.out.println();
        }
        System.out.println();
    }
}
