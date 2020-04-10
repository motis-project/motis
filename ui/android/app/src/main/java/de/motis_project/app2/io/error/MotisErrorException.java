package de.motis_project.app2.io.error;

public class MotisErrorException extends Exception {
    public final String category, reason;
    public final int code;

    public MotisErrorException(String category, String reason, int code) {
        this.category = category;
        this.reason = reason;
        this.code = code;
    }
}
