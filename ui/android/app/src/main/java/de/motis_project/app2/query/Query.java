package de.motis_project.app2.query;

import android.content.SharedPreferences;
import android.os.Bundle;

import java.util.Calendar;
import java.util.Date;

public class Query {
    private static final String IS_ARRIVAL = "IS_ARRIVAL";
    private static final String YEAR = "QUERY_YEAR";
    private static final String MONTH = "QUERY_MONTH";
    private static final String DAY = "QUERY_DAY";
    private static final String HOUR = "QUERY_HOUR";
    private static final String MINUTE = "QUERY_MINUTE";
    private static final String FROM_NAME = "QUERY_FROM_NAME";
    private static final String TO_NAME = "QUERY_TO_NAME";
    private static final String FROM_ID = "QUERY_FROM_ID";
    private static final String TO_ID = "QUERY_TO_ID";

    private final Bundle bundle;
    private final SharedPreferences pref;
    private final Calendar cal;

    public Query(Bundle b, SharedPreferences pref) {
        this.pref = pref;
        this.bundle = b != null ? b : new Bundle();
        this.cal = Calendar.getInstance();
    }

    public boolean isArrival() { return bundle.getBoolean(IS_ARRIVAL, false); }

    public int getYear() { return bundle.getInt(YEAR, cal.get(Calendar.YEAR)); }

    public int getMonth() { return bundle.getInt(MONTH, cal.get(Calendar.MONTH)); }

    public int getDay() { return bundle.getInt(DAY, cal.get(Calendar.DAY_OF_MONTH)); }

    public int getHour() { return bundle.getInt(HOUR, cal.get(Calendar.HOUR_OF_DAY)); }

    public int getMinute() { return bundle.getInt(MINUTE, cal.get(Calendar.MINUTE)); }

    public String getFromName() { return pref.getString(FROM_NAME, ""); }

    public String getToName() { return pref.getString(TO_NAME, ""); }

    public String getFromId() { return pref.getString(FROM_ID, ""); }

    public String getToId() { return pref.getString(TO_ID, ""); }

    public Date getTime() {
        Calendar cal = Calendar.getInstance();
        setDate(cal, getYear(), getMonth(), getDay());
        setTime(cal, getHour(), getMinute());
        return cal.getTime();
    }

    public static void setDate(Calendar cal, int year, int month, int day) {
        cal.set(Calendar.YEAR, year);
        cal.set(Calendar.MONTH, month);
        cal.set(Calendar.DAY_OF_MONTH, day);
    }

    public static void setTime(Calendar cal, int hour, int minute) {
        cal.set(Calendar.HOUR_OF_DAY, hour);
        cal.set(Calendar.MINUTE, minute);
    }

    public void setDate(int year, int month, int day) {
        bundle.putInt(YEAR, year);
        bundle.putInt(MONTH, month);
        bundle.putInt(DAY, day);
    }

    public void setTime(boolean isArrival, int hour, int minute) {
        bundle.putBoolean(IS_ARRIVAL, isArrival);
        bundle.putInt(HOUR, hour);
        bundle.putInt(MINUTE, minute);
    }

    public void setFrom(String id, String name) {
        SharedPreferences.Editor edit = pref.edit();
        edit.putString(FROM_ID, id);
        edit.putString(FROM_NAME, name);
        edit.apply();
    }

    public void setTo(String id, String name) {
        SharedPreferences.Editor edit = pref.edit();
        edit.putString(TO_ID, id);
        edit.putString(TO_NAME, name);
        edit.apply();
    }

    public void swapStations() {
        SharedPreferences.Editor edit = pref.edit();
        edit.putString(FROM_NAME, getToName());
        edit.putString(FROM_ID, getToId());
        edit.putString(TO_NAME, getFromName());
        edit.putString(TO_ID, getFromId());
        edit.apply();
    }

    public void updateBundle(Bundle outState) {
        outState.putInt(YEAR, getYear());
        outState.putInt(MONTH, getMonth());
        outState.putInt(DAY, getDay());
        outState.putBoolean(IS_ARRIVAL, isArrival());
        outState.putInt(HOUR, getHour());
        outState.putInt(MINUTE, getMinute());
    }

    public boolean isComplete() {
        return !getFromId().isEmpty() && !getToId().isEmpty() && !getFromId().equals(getToId());
    }
}