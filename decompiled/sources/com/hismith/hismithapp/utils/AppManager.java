package com.hismith.hismithapp.utils;

import android.app.Activity;
import java.util.Iterator;
import java.util.Stack;

/* loaded from: classes3.dex */
public class AppManager {
    private static AppManager sActivityManager;
    private static final Stack<Activity> sActivityStack = new Stack<>();

    private AppManager() {
    }

    public static AppManager getInstance() {
        AppManager appManager;
        AppManager appManager2 = sActivityManager;
        if (appManager2 != null) {
            return appManager2;
        }
        synchronized (AppManager.class) {
            appManager = new AppManager();
            sActivityManager = appManager;
        }
        return appManager;
    }

    public void AppExit() {
        try {
            finishAllActivity();
            System.exit(0);
        } catch (Exception unused) {
        }
    }

    public void addActivity(Activity activity) {
        sActivityStack.add(activity);
    }

    public void clearAllActivity() {
        Stack<Activity> stack = sActivityStack;
        if (stack != null) {
            stack.clear();
        }
    }

    public Activity currentActivity() {
        try {
            return sActivityStack.lastElement();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public void finishActivity() {
        finishActivity(sActivityStack.lastElement());
    }

    public void finishActivity(Activity activity) {
        if (activity != null) {
            sActivityStack.remove(activity);
            activity.finish();
        }
    }

    public void finishActivity(Class<?> cls) {
        Iterator<Activity> it = sActivityStack.iterator();
        while (it.hasNext()) {
            Activity next = it.next();
            if (next.getClass().equals(cls)) {
                finishActivity(next);
                return;
            }
        }
    }

    public void finishAllActivity() {
        int size = sActivityStack.size();
        for (int i = 0; i < size; i++) {
            Stack<Activity> stack = sActivityStack;
            if (stack.get(i) != null) {
                stack.get(i).finish();
            }
        }
        sActivityStack.clear();
    }

    public void finishAllOtherActivity(Activity activity) {
        int size = sActivityStack.size();
        for (int i = 0; i < size; i++) {
            Stack<Activity> stack = sActivityStack;
            if (stack.get(i) != null && stack.get(i) != activity) {
                stack.get(i).finish();
            }
        }
        sActivityStack.clear();
    }

    public void finishAllOtherActivity(Class<?> cls) {
        int size = sActivityStack.size();
        for (int i = 0; i < size; i++) {
            Stack<Activity> stack = sActivityStack;
            if (stack.get(i) != null && !stack.get(i).getClass().equals(cls)) {
                stack.get(i).finish();
            }
        }
        sActivityStack.clear();
    }

    public Activity getActivity(Class<?> cls) {
        Iterator<Activity> it = sActivityStack.iterator();
        while (it.hasNext()) {
            Activity next = it.next();
            if (next.getClass().equals(cls)) {
                return next;
            }
        }
        return null;
    }

    public Stack<Activity> getActivityStack() {
        return sActivityStack;
    }

    public void recreateAllActivity() {
        int size = sActivityStack.size();
        for (int i = 0; i < size; i++) {
            Stack<Activity> stack = sActivityStack;
            if (stack.get(i) != null) {
                stack.get(i).recreate();
            }
        }
    }

    public void recreateAllOtherActivity(Activity activity) {
        int size = sActivityStack.size();
        for (int i = 0; i < size; i++) {
            Stack<Activity> stack = sActivityStack;
            if (stack.get(i) != null && stack.get(i) != activity) {
                stack.get(i).recreate();
            }
        }
    }

    public void removeActivity(Activity activity) {
        Stack<Activity> stack = sActivityStack;
        if (stack.isEmpty()) {
            return;
        }
        stack.remove(activity);
    }
}
