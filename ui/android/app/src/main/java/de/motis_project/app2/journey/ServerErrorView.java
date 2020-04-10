package de.motis_project.app2.journey;

import android.content.Context;
import android.util.AttributeSet;
import android.widget.TextView;

import butterknife.BindString;
import butterknife.ButterKnife;
import de.motis_project.app2.R;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.io.error.MotisErrorException;
import motis.lookup.LookupScheduleInfoResponse;
import rx.Observable;
import rx.android.schedulers.AndroidSchedulers;
import rx.schedulers.Schedulers;

public class ServerErrorView extends TextView {
    @BindString(R.string.empty_response)
    String emptyResponseMessage;

    @BindString(R.string.schedule_range)
    String scheduleRangeTemplate;

    @BindString(R.string.routing_error)
    String routingErrorMessage;

    private final Observable<LookupScheduleInfoResponse> scheduleInfo = Status.get().getServer().scheduleInfo();

    public ServerErrorView(Context context) {
        super(context);
        ButterKnife.bind(this);
    }

    public ServerErrorView(Context context, AttributeSet attrs) {
        super(context, attrs);
        ButterKnife.bind(this);
    }

    public void setErrorCode(MotisErrorException mee) {
        setText(JourneyErrorViewHolder
                .buildMessage(mee, scheduleInfo.toBlocking().firstOrDefault(null), scheduleRangeTemplate, routingErrorMessage));
        scheduleInfo.subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(lookupScheduleInfoResponse ->
                        setText(JourneyErrorViewHolder.buildScheduleRangeError(lookupScheduleInfoResponse, scheduleRangeTemplate, routingErrorMessage)), Throwable::printStackTrace);
    }

    public void setEmptyResponse() {
        setText(emptyResponseMessage);
    }
}
