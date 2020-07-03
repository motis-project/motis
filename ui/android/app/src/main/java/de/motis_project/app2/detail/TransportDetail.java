package de.motis_project.app2.detail;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import butterknife.BindColor;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.Str;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.journey.ConnectionWrapper;
import motis.Connection;
import motis.EventInfo;
import motis.Stop;
import motis.Transport;

public class TransportDetail implements DetailViewHolder {
    private View layout;

    @BindView(R.id.detail_transport_dep_station) TextView station;
    @BindView(R.id.detail_transport_dep_time) TextView time;
    @BindView(R.id.detail_transpot_dep_delay) TextView delay;
    @BindView(R.id.detail_transport_direction_container) LinearLayout directionContainer;
    @BindView(R.id.detail_transport_direction) TextView direction;
    @BindView(R.id.detail_transport_vertline) View line;

    @BindColor(R.color.delayed) int colorRed;
    @BindColor(R.color.ontime) int colorGreen;

    protected final Stop stop;
    protected final DetailClickHandler clickHandler;

    TransportDetail(ConnectionWrapper conWrapper,
                    JourneyUtil.Section section,
                    ViewGroup parent,
                    LayoutInflater inflater) {
        clickHandler = (DetailClickHandler) inflater.getContext();
        layout = inflater.inflate(R.layout.detail_transport, parent, false);
        ButterKnife.bind(this, layout);

        Connection con = conWrapper.getConnection();
        stop = con.stops(section.from);
        String stopName = stop.station().name();
        if (stopName.equals("START")) {
            stopName = conWrapper.getStartName();
        } else if (stopName.equals("END")) {
            stopName = conWrapper.getDestinationName();
        }
        station.setText(stopName);

        time.setText(TimeUtil.formatTime(stop.departure().scheduleTime()));

        Transport transport = JourneyUtil.getTransport(con, section);
        long clasz = (transport != null) ? transport.clasz() : JourneyUtil.WALK_CLASS;
        JourneyUtil.setBackgroundColor(inflater.getContext(), line, clasz);

        String dir = getDirection(con, section);
        if (dir.isEmpty()) {
            directionContainer.setVisibility(View.GONE);
        } else {
            direction.setText(dir);
        }

        EventInfo dep = stop.departure();
        delay.setText(TimeUtil.delayString(dep));
        delay.setTextColor(TimeUtil.delay(dep) ? colorRed : colorGreen);
    }

    private static String getDirection(Connection con, JourneyUtil.Section s) {
        Transport transport = JourneyUtil.getTransport(con, s);
        return (transport == null) ? "" : Str.san(transport.direction());
    }

    @Override
    public View getView() {
        return layout;
    }

    @OnClick(R.id.detail_transport)
    public void onClick() {
        clickHandler.transportStopClicked(stop);
    }
}
