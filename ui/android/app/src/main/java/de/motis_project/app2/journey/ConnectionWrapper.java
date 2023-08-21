package de.motis_project.app2.journey;

import android.content.Context;

import androidx.core.content.ContextCompat;

import java.util.ArrayList;

import de.motis_project.app2.R;
import motis.Connection;
import motis.Stop;

public class ConnectionWrapper {
    private static final ArrayList<Integer> COLORS = new ArrayList<>();

    static {
        COLORS.add(R.color.md_blue500);
        COLORS.add(R.color.md_red500);
        COLORS.add(R.color.md_green500);
        COLORS.add(R.color.md_orange500);
        COLORS.add(R.color.md_purple500);
        COLORS.add(R.color.md_pink500);
        COLORS.add(R.color.md_indigo500);
        COLORS.add(R.color.md_teal500);
        COLORS.add(R.color.md_yellow500);
        COLORS.add(R.color.md_brown500);
    }

    private final Connection connection;
    private final int id;
    private String startNameOverride;
    private String destinationNameOverride;
    private boolean intermodal;

    public ConnectionWrapper(Connection connection, int id) {
        this.connection = connection;
        this.id = id;
    }

    public Connection getConnection() {
        return connection;
    }

    public int getId() {
        return id;
    }

    public int getColor(Context context) {
        int idx = id % COLORS.size();
        if (idx < 0) {
            idx += COLORS.size();
        }
        int colorId = COLORS.get(idx);
        return ContextCompat.getColor(context, colorId);
    }

    public Stop getFirstStop() {
        return connection.stops(0);
    }

    public Stop getLastStop() {
        return connection.stops(connection.stopsLength() - 1);
    }

    public String getStartName() {
        if (startNameOverride != null) {
            return startNameOverride;
        } else {
            return getFirstStop().station().name();
        }
    }

    public String getDestinationName() {
        if (destinationNameOverride != null) {
            return destinationNameOverride;
        } else {
            return getLastStop().station().name();
        }
    }

    public boolean isIntermodal() {
        return intermodal;
    }

    public void setStartNameOverride(String startNameOverride) {
        this.startNameOverride = startNameOverride;
    }

    public void setDestinationNameOverride(String destinationNameOverride) {
        this.destinationNameOverride = destinationNameOverride;
    }

    public int getNumberOfTransfers() {
        int transfers = 0;
        for (int i = 0; i < connection.stopsLength(); i++) {
            if (connection.stops(i).exit()) {
                transfers++;
            }
        }
        return transfers - 1;
    }

    @Override
    public String toString() {
        return "ConnectionWrapper{" +
                "connection=" + connection +
                ", id=" + id +
                ", startNameOverride='" + startNameOverride + '\'' +
                ", destinationNameOverride='" + destinationNameOverride + '\'' +
                '}';
    }
}
