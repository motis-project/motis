package de.motis_project.app2.ppr.query;

import android.content.SharedPreferences;

import de.motis_project.app2.ppr.NamedLocation;
import de.motis_project.app2.ppr.profiles.PprSearchOptions;
import de.motis_project.app2.ppr.profiles.SearchProfile;
import de.motis_project.app2.ppr.profiles.SearchProfiles;

public class PPRQuery {
    public NamedLocation placeFrom;
    public NamedLocation placeTo;
    public PprSearchOptions pprSearchOptions;
    public SearchProfiles profiles;

    public PPRQuery(SearchProfiles profiles) {
        this.profiles = profiles;
        this.pprSearchOptions = (profiles != null) ? profiles.getDefaultSearchOptions() : null;
    }

    public void swapStartDest() {
        NamedLocation oldFrom = placeFrom;
        placeFrom = placeTo;
        placeTo = oldFrom;
    }

    public boolean isComplete() {
        return placeFrom != null && placeTo != null;
    }

    public void save(SharedPreferences prefs, String prefix) {
        SharedPreferences.Editor editor = prefs.edit();
        saveLocation(editor, prefix + "placeFrom.", placeFrom);
        saveLocation(editor, prefix + "placeTo.", placeTo);
        if (pprSearchOptions != null) {
            editor.putString(prefix + "pprSearchOptions.profile.id", pprSearchOptions.profile.getId());
            editor.putInt(prefix + "pprSearchOptions.maxDuration", pprSearchOptions.maxDuration);
        }
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

    public void load(SharedPreferences prefs, String prefix) {
        placeFrom = loadLocation(prefs, prefix + "placeFrom.", placeFrom);
        placeTo = loadLocation(prefs, prefix + "placeTo.", placeTo);

        String defaultProfile = pprSearchOptions != null ? pprSearchOptions.profile.getId() : "default";
        int defaultDuration =
                pprSearchOptions != null ? pprSearchOptions.maxDuration : SearchProfiles.DEFAULT_MAX_DURATION;
        SearchProfile searchProfile = profiles.getById(
                prefs.getString(prefix + "pprSearchOptions.profile.id", defaultProfile), profiles.getDefault());
        int maxDuration = prefs.getInt(prefix + "pprSearchOptions.maxDuration", defaultDuration);
        pprSearchOptions = new PprSearchOptions(searchProfile, maxDuration);
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
}
