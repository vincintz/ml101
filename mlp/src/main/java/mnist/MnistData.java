package mnist;

import java.io.*;

public class MnistData {
    private final int numberOfItems;
    private final int rows;
    private final int cols;
    private final byte[][] input;
    private final byte[] output;

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

            // read labels
            output = new byte[numberOfItems];
            labels.read(output);

            // read images
            rows = images.readInt();
            cols = images.readInt();
            int imageSize = rows * cols;
            input = new byte[numberOfItems][];
            for (int i = 0; i < numberOfItems; i++) {
                input[i] = new byte[imageSize];
                images.read(input[i]);
            }
        }
        catch (final IOException ex) {
            throw new IllegalArgumentException("Can't read file", ex);
        }
    }

    public int numberOfItems() {
        return numberOfItems;
    }
}
