package com.hismith.hismithapp.utils;

import android.view.View;
import io.reactivex.Observable;
import io.reactivex.ObservableEmitter;
import io.reactivex.ObservableOnSubscribe;
import io.reactivex.functions.Consumer;
import java.util.concurrent.TimeUnit;

/* loaded from: classes3.dex */
public class RxView {

    public interface Action1<T> {
        void onClick(T t);
    }

    private static class ViewClickOnSubscribe implements ObservableOnSubscribe<View> {
        private View view;

        public ViewClickOnSubscribe(View view) {
            this.view = view;
        }

        @Override // io.reactivex.ObservableOnSubscribe
        public void subscribe(final ObservableEmitter<View> observableEmitter) throws Exception {
            Preconditions.checkUiThread();
            this.view.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.RxView.ViewClickOnSubscribe.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (observableEmitter.isDisposed()) {
                        return;
                    }
                    observableEmitter.onNext(ViewClickOnSubscribe.this.view);
                }
            });
        }
    }

    private static Observable<View> onClick(View view) {
        Preconditions.checkNotNull(view, "view == null");
        return Observable.create(new ViewClickOnSubscribe(view));
    }

    public static void setOnClickListeners(final Action1<View> action1, View... viewArr) {
        for (View view : viewArr) {
            onClick(view).throttleFirst(500L, TimeUnit.MILLISECONDS).subscribe(new Consumer<View>() { // from class: com.hismith.hismithapp.utils.RxView.1
                @Override // io.reactivex.functions.Consumer
                public void accept(View view2) throws Exception {
                    Action1.this.onClick(view2);
                }
            });
        }
    }
}
