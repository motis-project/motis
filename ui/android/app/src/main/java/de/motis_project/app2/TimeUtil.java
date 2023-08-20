package de.motis_project.app2;

import androidx.annotation.NonNull;

import java.text.SimpleDateFormat;
import java.util.Date;

import motis.EventInfo;
import motis.TimestampReason;

public class TimeUtil {
    static StringBuffer durationBuf = new StringBuffer();

    @NonNull
    public static String formatDuration(long minutes) {
        durationBuf.setLength(0);

        long displayMinutes = minutes % 60;
        long displayHours = minutes / 60;

        if (displayHours != 0) {
            durationBuf.append(displayHours).append("h ");
        }
        if (displayMinutes != 0) {
            durationBuf.append(displayMinutes).append("min");
        }

        return durationBuf.toString();
    }

    public static String formatTime(long unixTimestamp) {
        return formatTime(new Date(unixTimestamp * 1000));
    }

    public static String formatTime(Date time) {
        return SimpleDateFormat
                .getTimeInstance(java.text.DateFormat.SHORT)
                .format(time);
    }

    public static String formatDate(long unixTimestamp) {
        return formatDate(new Date(unixTimestamp * 1000));
    }

    public static String formatDate(Date date) {
        return SimpleDateFormat
                .getDateInstance(java.text.DateFormat.SHORT)
                .format(date);
    }

    public static String delayString(EventInfo ev) {
        if (ev.reason() == TimestampReason.SCHEDULE) {
            return "";
        }

        long diffSeconds = ev.time() - ev.scheduleTime();
        long diffMinutes = diffSeconds / 60;
        if (diffMinutes >= 0) {
            return "+" + Long.toString(diffMinutes);
        } else {
            return Long.toString(diffMinutes);
        }
    }

    public static boolean delay(EventInfo ev) {
        return (ev.time() - ev.scheduleTime()) > 0;
    }
}
