package de.motis_project.app2.intermodal;

import android.content.SharedPreferences;

import java.util.Calendar;

import de.motis_project.app2.ppr.NamedLocation;
import de.motis_project.app2.ppr.profiles.PprSearchOptions;
import de.motis_project.app2.ppr.profiles.SearchProfile;
import de.motis_project.app2.ppr.profiles.SearchProfiles;

public class IntermodalQuery {
    private NamedLocation placeFrom;
    private NamedLocation placeTo;
    private Calendar dateTime;
    private boolean arrival;
    private PPRSettings pprSettings;

    public IntermodalQuery(SearchProfiles pprSearchProfiles) {
        this.dateTime = Calendar.getInstance();
        this.pprSettings = new PPRSettings(pprSearchProfiles);
    }

    public void swapStartDest() {
        NamedLocation oldFrom = placeFrom;
        placeFrom = placeTo;
        placeTo = oldFrom;
    }

    public boolean isComplete() {
        return placeFrom != null && placeTo != null;
    }

    public NamedLocation getPlaceFrom() {
        return placeFrom;
    }

    public void setPlaceFrom(NamedLocation placeFrom) {
        this.placeFrom = placeFrom;
    }

    public NamedLocation getPlaceTo() {
        return placeTo;
    }

    public void setPlaceTo(NamedLocation placeTo) {
        this.placeTo = placeTo;
    }

    public boolean isArrival() {
        return arrival;
    }

    public void setArrival(boolean arrival) {
        this.arrival = arrival;
    }

    public Calendar getDateTime() {
        return dateTime;
    }

    public void setDateTime(Calendar dateTime) {
        this.dateTime = dateTime;
    }

    public PPRSettings getPPRSettings() {
        return pprSettings;
    }

    public void setPPRSettings(PPRSettings pprSettings) {
        this.pprSettings = pprSettings;
    }

    public void setDate(int year, int month, int day) {
        dateTime.set(Calendar.YEAR, year);
        dateTime.set(Calendar.MONTH, month);
        dateTime.set(Calendar.DAY_OF_MONTH, day);
    }

    public void setTime(int hour, int minute) {
        dateTime.set(Calendar.HOUR_OF_DAY, hour);
        dateTime.set(Calendar.MINUTE, minute);
    }

    public void setTime(boolean arrival, int hour, int minute) {
        this.arrival = arrival;
        setTime(hour, minute);
    }

    public int getYear() {
        return dateTime.get(Calendar.YEAR);
    }

    public int getMonth() {
        return dateTime.get(Calendar.MONTH);
    }

    public int getDay() {
        return dateTime.get(Calendar.DAY_OF_MONTH);
    }

    public int getHour() {
        return dateTime.get(Calendar.HOUR_OF_DAY);
    }

    public int getMinute() {
        return dateTime.get(Calendar.MINUTE);
    }

    public void save(SharedPreferences prefs, String prefix) {
        SharedPreferences.Editor editor = prefs.edit();
        saveLocation(editor, prefix + "placeFrom.", placeFrom);
        saveLocation(editor, prefix + "placeTo.", placeTo);
        saveCalendar(editor, prefix + "dateTime.", dateTime);
        editor.putBoolean(prefix + "arrival", arrival);
        pprSettings.save(editor, prefix + "ppr.");
        editor.apply();
    }

    private void saveLocation(
            SharedPreferences.Editor editor, String prefix, NamedLocation location) {
        if (location != null) {
            editor.putString(prefix + "name", location.name);
            editor.putString(prefix + "lat", Double.toString(location.lat));
            editor.putString(prefix + "lng", Double.toString(location.lng));
        }
    }

    private void saveCalendar(SharedPreferences.Editor editor, String prefix, Calendar cal) {
        editor.putLong(prefix + ".millis", cal.getTimeInMillis());
    }

    public void load(SharedPreferences prefs, String prefix) {
        placeFrom = loadLocation(prefs, prefix + "placeFrom.", placeFrom);
        placeTo = loadLocation(prefs, prefix + "placeTo.", placeTo);
        dateTime = loadCalendar(prefs, prefix + "dateTime.", dateTime);
        arrival = prefs.getBoolean(prefix + "arrival", arrival);
        pprSettings.load(prefs, prefix + "ppr.");
    }

    private NamedLocation loadLocation(SharedPreferences prefs, String prefix, NamedLocation def) {
        String name = prefs.getString(prefix + "name", null);
        String latStr = prefs.getString(prefix + "lat", null);
        String lngStr = prefs.getString(prefix + "lng", null);
        if (name != null && latStr != null && lngStr != null) {
            double lat = Double.parseDouble(latStr);
            double lng = Double.parseDouble(lngStr);
            return new NamedLocation(name, lat, lng);
        }
        return def;
    }

    private Calendar loadCalendar(SharedPreferences prefs, String prefix, Calendar def) {
        long ts = prefs.getLong(prefix + ".millis", 0);
        if (ts != 0) {
            Calendar cal = Calendar.getInstance();
            cal.setTimeInMillis(ts);
            return cal;
        } else {
            return def;
        }
    }

    public static class PPRSettings {
        public PprSearchOptions pprSearchOptions;
        public final SearchProfiles profiles;

        public PPRSettings(SearchProfiles profiles) {
            this.profiles = profiles;
            this.pprSearchOptions = profiles.getDefaultSearchOptions();
        }

        public void load(SharedPreferences prefs, String prefix) {
            String defaultProfile = pprSearchOptions != null ? pprSearchOptions.profile.getId() : "default";
            int defaultDuration =
                    pprSearchOptions != null ? pprSearchOptions.maxDuration : SearchProfiles.DEFAULT_MAX_DURATION;
            SearchProfile searchProfile = profiles.getById(
                    prefs.getString(prefix + "pprSearchOptions.profile.id", defaultProfile), profiles.getDefault());
            int maxDuration = prefs.getInt(prefix + "pprSearchOptions.maxDuration", defaultDuration);
            pprSearchOptions = new PprSearchOptions(searchProfile, maxDuration);
        }

        public void save(SharedPreferences.Editor editor, String prefix) {
            if (pprSearchOptions != null) {
                editor.putString(prefix + "pprSearchOptions.profile.id", pprSearchOptions.profile.getId());
                editor.putInt(prefix + "pprSearchOptions.maxDuration", pprSearchOptions.maxDuration);
            }
        }

        public void setProfile(SearchProfile profile) {
            pprSearchOptions.profile = profile;
        }
    }
}
