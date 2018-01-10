#include "GTKFrontend.h"

#if GTK_ENABLED

#include <iostream>

GTKFrontend::GTKFrontend(ApplicationOptions const &options) {
    int argc = 0;
    char **argv = nullptr;
    app = Gtk::Application::create(argc, argv, "pl.edu.uj.tcs.raytracer");

    Glib::RefPtr<Gtk::Builder> builder = Gtk::Builder::create_from_file(
            "MainWindow.glade");

    builder->get_widget("mainWindow", mainWindow);
    builder->get_widget("headerBar", headerBar);
    builder->get_widget("contentStack", contentStack);
    builder->get_widget("renderedImageSpinner", renderedImageSpinner);
    builder->get_widget("renderedImage", renderedImage);

    renderedImage->signal_draw().connect(
            sigc::mem_fun(*this, &GTKFrontend::drawImage));

    mainWindow->resize(
            options.width / renderedImage->get_scale_factor(),
            options.height / renderedImage->get_scale_factor());

    createActions();
    dispatcher.connect(sigc::mem_fun(*this, &GTKFrontend::onImageSet));
}

void GTKFrontend::createActions() {
    const Glib::RefPtr<Gio::SimpleActionGroup> &refActionGroup =
            Gio::SimpleActionGroup::create();

    refreshAction = refActionGroup->add_action(
            "refresh", sigc::mem_fun(*this, &GTKFrontend::onRefresh));
    refreshAction->set_enabled(false);
    saveAction = refActionGroup->add_action(
            "save", sigc::mem_fun(*this, &GTKFrontend::onSave));
    saveAction->set_enabled(false);

    mainWindow->insert_action_group("app", refActionGroup);
}

GTKFrontend::~GTKFrontend() {
    // Only the main window needs to be deleted, as the children are managed
    // and removed automatically with their parent
    delete mainWindow;
}

void GTKFrontend::run() {
    app->run(*mainWindow);
}

void GTKFrontend::setImage(Bitmap image) {
    std::unique_lock<std::mutex> localImageLock(imageLock);
    newBitmap = std::make_unique<Bitmap>(std::move(image));
    dispatcher.emit();
}

void GTKFrontend::onImageSet() {
    std::unique_lock<std::mutex> localImageLock(imageLock);

    bitmapPixbuf = std::move(Gdk::Pixbuf::create_from_data(
            newBitmap->pixelData, Gdk::COLORSPACE_RGB, false, 8, newBitmap->width,
            newBitmap->height, newBitmap->width * newBitmap->bytesPerPixel));

    std::swap(bitmap, newBitmap);
    newBitmap = nullptr;

    headerBar->set_subtitle(
            std::to_string(bitmap->width) + " x " +
            std::to_string(bitmap->height));
    refreshAction->set_enabled(true);
    saveAction->set_enabled(true);
    contentStack->set_visible_child(*renderedImage);
}

bool GTKFrontend::drawImage(Cairo::RefPtr<Cairo::Context> const &context) {
    int scaleFactor = renderedImage->get_scale_factor();
    int width = renderedImage->get_allocated_width() * scaleFactor;
    int height = renderedImage->get_allocated_height() * scaleFactor;

    const Glib::RefPtr<Gdk::Pixbuf> &img = bitmapPixbuf->scale_simple(
            width, height, Gdk::InterpType::INTERP_BILINEAR);

    context->scale(1. / scaleFactor, 1. / scaleFactor);
    Gdk::Cairo::set_source_pixbuf(context, img, 0, 0);
    context->paint();
}

void GTKFrontend::onRefresh() {
    contentStack->set_visible_child(*renderedImageSpinner);
    headerBar->set_subtitle("");
    refreshAction->set_enabled(false);
    saveAction->set_enabled(false);

    backendController->setResolution(
            static_cast<unsigned>(renderedImage->get_width() *
                    renderedImage->get_scale_factor()),
            static_cast<unsigned>(renderedImage->get_height() *
                    renderedImage->get_scale_factor()));
    backendController->refresh();
}

void GTKFrontend::onSave() {
    Gtk::FileChooserDialog dialog(
            "Save the image", Gtk::FILE_CHOOSER_ACTION_SAVE);
    dialog.set_transient_for(*mainWindow);

    dialog.add_button("_Cancel", Gtk::RESPONSE_CANCEL);
    dialog.add_button("_Save", Gtk::RESPONSE_OK);

    auto filterPNG = Gtk::FileFilter::create();
    filterPNG->set_name("PNG images");
    filterPNG->add_pattern("*.png");
    filterPNG->add_mime_type("image/png");
    dialog.add_filter(filterPNG);

    auto filterJPEG = Gtk::FileFilter::create();
    filterJPEG->set_name("JPEG images");
    filterJPEG->add_mime_type("image/jpeg");
    dialog.add_filter(filterJPEG);

    auto filterBMP = Gtk::FileFilter::create();
    filterBMP->set_name("BMP images");
    filterBMP->add_mime_type("image/bmp");
    dialog.add_filter(filterBMP);

    if (dialog.run() == Gtk::RESPONSE_OK) {
        std::string filename = dialog.get_filename();

        std::string type;
        auto filter = dialog.get_filter();
        if (filter == filterPNG) {
            type = "png";
        } else if (filter == filterJPEG) {
            type = "jpeg";
        } else if (filter == filterBMP) {
            type = "bmp";
        }

        bitmapPixbuf->save(dialog.get_filename(), type);
    }
}

#endif //GTK_ENABLED
