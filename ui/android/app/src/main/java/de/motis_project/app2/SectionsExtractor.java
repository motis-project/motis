package de.motis_project.app2;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;

import motis.Connection;

public class SectionsExtractor {
    static public List<JourneyUtil.Section> getSections(Connection con, boolean includeIntermodalWalks) {
        List<JourneyUtil.Section> sections = new ArrayList<>();
        if (includeIntermodalWalks && con.stopsLength() > 1 && !con.stops(0).enter()) {
            sections.add(new JourneyUtil.Section(0, 1));
        }
        for (int i = 0; i < con.stopsLength(); ++i) {
            if (con.stops(i).enter()) {
                sections.add(new JourneyUtil.Section(i, getNextExit(con, i)));
            }
        }
        if (includeIntermodalWalks && con.stopsLength() > 1 && !con.stops(con.stopsLength() - 1).exit()) {
            sections.add(new JourneyUtil.Section(con.stopsLength() - 2, con.stopsLength() - 1));
        }
        return sections;
    }

    static private int getNextExit(Connection con, int stopIndex) {
        for (int i = stopIndex + 1; i < con.stopsLength(); i++) {
            if (con.stops(i).exit()) {
                return i;
            }
        }
        Log.e("MOTIS", "section end not found");
        return con.stopsLength() - 1;
    }
}
