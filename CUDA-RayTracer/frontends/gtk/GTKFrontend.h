#ifndef RAY_TRACER_GTKFRONTEND_H
#define RAY_TRACER_GTKFRONTEND_H

#include "application/CompileSettings.h"

#if GTK_ENABLED

#include <gtkmm.h>

#include "frontends/Frontend.h"
#include "application/ApplicationOptions.h"
#include "backends/Image.h"

class GTKFrontend : public Frontend {
private:
    Glib::RefPtr<Gtk::Application> app;

    Gtk::Window *mainWindow;
    Gtk::HeaderBar *headerBar;
    Gtk::Stack *contentStack;
    Gtk::Spinner *renderedImageSpinner;
    Gtk::DrawingArea *renderedImage;

    Glib::RefPtr<Gio::SimpleAction> refreshAction;
    Glib::RefPtr<Gio::SimpleAction> saveAction;

    Glib::Dispatcher dispatcher;

    std::mutex imageLock;
    std::unique_ptr<Bitmap> bitmap;
    std::unique_ptr<Bitmap> newBitmap;
    Glib::RefPtr<Gdk::Pixbuf> bitmapPixbuf;

    void createActions();

    void onImageSet();

    bool drawImage(Cairo::RefPtr<Cairo::Context> const &context);

    void onRefresh();

    void onSave();
public:
    explicit GTKFrontend(ApplicationOptions const &options);

    ~GTKFrontend() override;

    void run() override;

    void setImage(Bitmap image) override;
};

#endif //GTK_ENABLED
#endif //RAY_TRACER_GTKFRONTEND_H
