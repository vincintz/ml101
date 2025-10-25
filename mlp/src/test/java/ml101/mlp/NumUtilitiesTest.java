package ml101.mlp;

import ml101.mlp.activation.ActivationFn;
import org.junit.Test;

import static org.junit.Assert.*;

public class NumUtilitiesTest {

    private static final double EPS = 1e-9;

    @Test
    public void testCrossMultiply() {
        double[][] matrix = new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        };
        double[] vector = new double[]{10.0, 100.0};
        double[] result = new double[2];

        NumUtilities.crossMultiply(result, matrix, vector);

        assertArrayEquals(new double[]{210.0, 430.0}, result, EPS);
    }

    @Test
    public void testVectorAddWithDifferentLengths() {
        double[] v1 = new double[]{1.0, 2.0, 3.0};
        double[] v2 = new double[]{4.0, 5.0};
        double[] result = new double[]{-1.0, -1.0, -1.0};

        NumUtilities.vectorAdd(result, v1, v2);

        // Only first two positions should be updated (min length = 2)
        assertEquals(5.0, result[0], EPS);
        assertEquals(7.0, result[1], EPS);
        // Third position should remain unchanged
        assertEquals(-1.0, result[2], EPS);
    }

    @Test
    public void testZerosFrom2D() {
        double[][] shape = new double[][]{
                {1.0, 2.0, 3.0},
                {4.0}
        };
        double[][] zeros = NumUtilities.zerosFrom(shape);

        assertNotNull(zeros);
        assertEquals(2, zeros.length);
        assertEquals(3, zeros[0].length);
        assertEquals(1, zeros[1].length);

        for (int i = 0; i < zeros.length; i++) {
            for (int j = 0; j < zeros[i].length; j++) {
                assertEquals(0.0, zeros[i][j], EPS);
            }
        }
    }

    @Test
    public void testZerosFrom3D() {
        double[][][] shape = new double[2][][];
        shape[0] = new double[][]{{1.0, 2.0}, {3.0}};
        shape[1] = new double[][]{{4.0}};

        double[][][] zeros = NumUtilities.zerosFrom(shape);

        assertNotNull(zeros);
        assertEquals(2, zeros.length);
        assertEquals(2, zeros[0].length);
        assertEquals(1, zeros[1].length);
        assertEquals(2, zeros[0][0].length);
        assertEquals(1, zeros[0][1].length);
        assertEquals(1, zeros[1][0].length);

        for (int l = 0; l < zeros.length; l++) {
            for (int j = 0; j < zeros[l].length; j++) {
                for (int i = 0; i < zeros[l][j].length; i++) {
                    assertEquals(0.0, zeros[l][j][i], EPS);
                }
            }
        }
    }

    @Test
    public void testActivateAndSquare() {
        double[] v = new double[]{1.0, 2.0, 3.0};

        ActivationFn dbl = new ActivationFn() {
            @Override
            public double compute(double z) {
                return z * 2.0;
            }

            @Override
            public double derivative(double z) {
                return 2.0;
            }
        };

        NumUtilities.activate(v, dbl);
        assertArrayEquals(new double[]{2.0, 4.0, 6.0}, v, EPS);

        assertEquals(25.0, NumUtilities.square(-5.0), EPS);
    }
}
