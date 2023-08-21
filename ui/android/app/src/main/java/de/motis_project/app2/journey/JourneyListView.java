package de.motis_project.app2.journey;

import android.content.Context;
import android.content.res.Resources;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.io.error.MotisErrorException;
import de.motis_project.app2.lib.SimpleDividerItemDecoration;
import de.motis_project.app2.lib.StickyHeaderDecoration;
import de.motis_project.app2.query.Query;
import motis.Connection;
import motis.routing.RoutingResponse;
import rx.Subscription;
import rx.android.schedulers.AndroidSchedulers;
import rx.functions.Action1;
import rx.internal.util.SubscriptionList;
import rx.schedulers.Schedulers;

public class JourneyListView
        extends BaseJourneyListView<Query> {

    public JourneyListView(Context context) {
        super(context);
    }

    public JourneyListView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public JourneyListView(Context context, @Nullable AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
    }
}
