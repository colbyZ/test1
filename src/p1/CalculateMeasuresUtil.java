package p1;

import java.util.ArrayList;
import java.util.List;

public final class CalculateMeasuresUtil {

  private static List<Double> getMotherAgeList(double[][] birthWeightTable, int numberOfRows, double minInfantWeight, double maxInfantWeight) {
    List<Double> motherAgeList = new ArrayList<>();
    int size = Math.min(birthWeightTable.length, numberOfRows);
    for (int i = 0; i < size; i++) {
      double[] row = birthWeightTable[i];
      double motherAge = row[0];
      double infantWeight = row[1];
      if (minInfantWeight <= infantWeight && infantWeight <= maxInfantWeight) {
        motherAgeList.add(motherAge);
      }
    }
    return motherAgeList;
  }

  private static double calculateAverage(List<Double> doubleList) {
    double sum = 0.0;
    for (Double d : doubleList) {
      sum += d;
    }
    return sum / doubleList.size();
  }

  public static Measures calculateMeasures(double[][] birthWeightTable, int numberOfRows, double minInfantWeight, double maxInfantWeight) {
    double average;
    double median;
    double mode;
    List<Double> motherAgeList = getMotherAgeList(birthWeightTable, numberOfRows, minInfantWeight, maxInfantWeight);
    if (motherAgeList.isEmpty()) {
      average = Double.NaN;
      median = Double.NaN;
      mode = Double.NaN;
    } else {
      average = calculateAverage(motherAgeList);
      median = 0;
      mode = calculateMode(motherAgeList);
    }
    return new Measures(average, median, mode);
  }

  private static double calculateMode(List<Double> motherAgeList) {
    return 0;
  }

  public static final class Measures {

    public final double average;
    public final double median;
    public final double mode;

    private Measures(double average, double median, double mode) {
      this.average = average;
      this.median = median;
      this.mode = mode;
    }

  }

  private CalculateMeasuresUtil() {
  }

}
