package p1.test;

import p1.CalculateMeasuresUtil;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public final class CalculateMeasuresUtilTest {

  private static void assertDoubleEquals(double expected, double actual) {
    assertEquals(expected, actual, 1e-6);
  }

  @Test
  public void simple() {
    double[][] birthWeightTable = {
        {24.0, 4.5}
    };
    CalculateMeasuresUtil.Measures measures = CalculateMeasuresUtil.calculateMeasures(birthWeightTable, birthWeightTable.length, 0.0, 5.0);
    assertDoubleEquals(measures.average, 24.0);
    assertDoubleEquals(measures.mode, 24.0);
  }

}
