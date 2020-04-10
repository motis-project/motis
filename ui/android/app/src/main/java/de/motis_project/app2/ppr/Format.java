package de.motis_project.app2.ppr;

public class Format {
    public static String formatDistance(double meters) {
        if (meters >= 1000) {
            return String.format("%.2fkm", meters / 1000);
        } else {
            return String.format("%.0fm", meters);
        }
    }
}
