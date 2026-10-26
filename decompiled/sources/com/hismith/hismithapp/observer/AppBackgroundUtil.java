package com.hismith.hismithapp.observer;

import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class AppBackgroundUtil {
    public static boolean isBackground = true;
    private static List<Observer> observerList = new ArrayList();

    public interface Observer {
        void onChange(boolean z);
    }

    public static void clearObserver() {
        observerList.clear();
    }

    public static void dispatchBackgroundStatus(boolean z) {
        for (int i = 0; i < observerList.size(); i++) {
            observerList.get(i).onChange(z);
        }
    }

    public static boolean isAppBackground() {
        return isBackground;
    }

    public static void observe(Observer observer) {
        observerList.add(observer);
    }

    public static int observeSize() {
        List<Observer> list = observerList;
        if (list != null) {
            return list.size();
        }
        return 0;
    }

    public static void stopObserve(Observer observer) {
        observerList.remove(observer);
    }
}
