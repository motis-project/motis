package de.motis_project.app2.detail;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import butterknife.BindString;
import butterknife.BindView;
import butterknife.ButterKnife;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import motis.Connection;
import motis.EventInfo;

public class TransportHeader implements DetailViewHolder {
    private View layout;

    @BindString(R.string.arrival_short) String arrivalShort;
    @BindString(R.string.interchange) String interchange;
    @BindString(R.string.walk) String walk;
    @BindString(R.string.track_short) String trackShort;
    @BindString(R.string.track) String track;

    @BindView(R.id.detail_transport_header_transport_name) TextView transportName;
    @BindView(R.id.detail_transport_header_interchange) TextView interchangeInfo;
    @BindView(R.id.detail_transport_header_track) TextView depTrack;

    TransportHeader(Connection con,
                    JourneyUtil.Section prevSection,
                    JourneyUtil.Section section,
                    ViewGroup parent,
                    LayoutInflater inflater) {
        layout = inflater.inflate(R.layout.detail_transport_header, parent, false);
        ButterKnife.bind(this, layout);

        Context context = inflater.getContext();
        long clasz = JourneyUtil.getTransport(con, section).clasz();
        JourneyUtil.tintBackground(context, transportName, clasz);
        JourneyUtil.setIcon(context, transportName, clasz);

        transportName.setText(JourneyUtil.getTransportName(con, section));

        EventInfo arr = con.stops(prevSection.to).arrival();
        EventInfo dep = con.stops(section.from).departure();
        EventInfo walkArr = con.stops(section.from).arrival();
        long arrTime = arr.time();
        long depTime = dep.time();
        long walkArrTime = walkArr.time();
        String arrTrackName = arr.track();

        boolean isWalk = prevSection.to != section.from;

        long duration = ((isWalk ? walkArrTime : depTime) - arrTime) / 60;
        String durationStr = TimeUtil.formatDuration(duration);

        if (arrTrackName == null || arrTrackName.isEmpty()) {
            interchangeInfo.setText(String.format(isWalk ? walk : interchange, durationStr));
        } else {
            arrTrackName = arrivalShort + " " + trackShort + " " + arrTrackName;
            System.out.println(durationStr);
            interchangeInfo.setText(
                    String.format(isWalk ? walk : interchange, arrTrackName + ", " + durationStr));
        }

        String depTrackName = dep.track();
        if (depTrackName == null || depTrackName.isEmpty()) {
            depTrack.setVisibility(View.GONE);
        } else {
            depTrack.setText(String.format(track, depTrackName));
        }
    }

    @Override
    public View getView() {
        return layout;
    }
}
