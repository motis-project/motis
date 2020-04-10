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
import de.motis_project.app2.Str;
import motis.Connection;

public class FirstTransportHeader implements DetailViewHolder {
    private View layout;

    @BindString(R.string.track) String track;

    @BindView(R.id.detail_first_transport_name) TextView transportName;
    @BindView(R.id.detail_first_transport_track) TextView depTrack;

    FirstTransportHeader(Connection con,
                         JourneyUtil.Section section,
                         ViewGroup parent,
                         LayoutInflater inflater) {
        layout = inflater.inflate(
                R.layout.detail_first_transport_header, parent, false);
        ButterKnife.bind(this, layout);

        transportName.setText(JourneyUtil.getTransportName(con, section));

        Context context = inflater.getContext();
        long clasz = JourneyUtil.getTransport(con, section).clasz();
        JourneyUtil.tintBackground(context, transportName, clasz);
        JourneyUtil.setIcon(context, transportName, clasz);

        String trackName = Str.san(con.stops(section.from).departure().track());
        if (trackName.isEmpty()) {
            depTrack.setVisibility(View.GONE);
        } else {
            depTrack.setText(String.format(track, trackName));
        }
    }

    @Override
    public View getView() {
        return layout;
    }
}
