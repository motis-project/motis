package de.motis_project.app2.detail;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import butterknife.BindColor;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.journey.ConnectionWrapper;
import motis.Connection;
import motis.EventInfo;
import motis.Stop;
import motis.Transport;

public class TransportTargetStation implements DetailViewHolder {
    private View layout;

    @BindView(R.id.detail_transport_target_station_arr_time) TextView arrivalTime;
    @BindView(R.id.detail_transport_target_station_delay) TextView arrivalDelay;
    @BindView(R.id.detail_transport_target_station) TextView targetStation;
    @BindView(R.id.detail_transport_target_station_vertline) View line;

    @BindColor(R.color.delayed) int colorRed;
    @BindColor(R.color.ontime) int colorGreen;

    protected final Stop stop;
    protected final DetailClickHandler clickHandler;

    TransportTargetStation(ConnectionWrapper conWrapper,
                           JourneyUtil.Section section,
                           ViewGroup parent,
                           LayoutInflater inflater) {
        clickHandler = (DetailClickHandler) inflater.getContext();
        layout = inflater.inflate(R.layout.detail_transport_target_stop, parent, false);
        ButterKnife.bind(this, layout);

        Connection con = conWrapper.getConnection();
        Transport transport = JourneyUtil.getTransport(con, section);
        long clasz = (transport != null) ? transport.clasz() : JourneyUtil.WALK_CLASS;
        JourneyUtil.setBackgroundColor(inflater.getContext(), line, clasz);

        stop = con.stops(section.to);

        arrivalTime.setText(TimeUtil.formatTime(stop.arrival().scheduleTime()));
        String stopName = stop.station().name();
        if (stopName.equals("START")) {
            stopName = conWrapper.getStartName();
        } else if (stopName.equals("END")) {
            stopName = conWrapper.getDestinationName();
        }
        targetStation.setText(stopName);

        EventInfo arr = stop.arrival();
        arrivalDelay.setText(TimeUtil.delayString(arr));
        arrivalDelay.setTextColor(TimeUtil.delay(arr) ? colorRed : colorGreen);
        arrivalDelay.setVisibility(View.VISIBLE);
    }

    @Override
    public View getView() {
        return layout;
    }

    @OnClick(R.id.detail_transport_target_stop)
    public void onClick() {
        clickHandler.transportStopClicked(stop);
    }
}
