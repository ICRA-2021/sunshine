//
// Created by stewart on 2020-09-30.
//

#ifndef _THREAD_POOL_HPP_
#define _THREAD_POOL_HPP_

#include <thread>
#include <list>

namespace sunshine {
    class thread_pool {
        std::list<std::pair<std::thread, std::unique_ptr<bool>>> threads;
        size_t const _size;
        std::atomic<size_t> active;
        std::mutex lock;
        std::condition_variable cv;
        bool waiting = false;

      public:
        explicit thread_pool(size_t const size = std::thread::hardware_concurrency())
            : _size(size), active(0) {
            if (size == 0) throw std::invalid_argument("Must have at least one thread");
        }

        size_t size() const {
            return _size;
        }

        void cleanup() {
            std::lock_guard<std::mutex> lk(lock);
            if (waiting) return;
            auto thread_iter = threads.begin();
            while (thread_iter != threads.end()) {
                assert(thread_iter->second);
                if (*(thread_iter->second)) {
                    // thread is finished, we can keep the lock
                    thread_iter->first.join();
                    thread_iter = threads.erase(thread_iter);
                } else {
                    thread_iter++;
                }
            }
        }

        template <typename Function>
        void enqueue(Function f) {
            {
                auto finished = std::make_unique<bool> (false);
                std::lock_guard<std::mutex> guard (lock);
                threads.emplace_back (std::make_pair (std::thread ([this, finishedPtr = finished.get (), func = std::move (f)] () {
                    assert(*finishedPtr == false);
                    bool ready = false;
                    do
                    {
                        size_t n_active = active;
                        while (n_active < _size)
                        {
                            ready = true;
                            if (active.compare_exchange_weak (n_active, n_active + 1)) break;
                            ready = false;
                        }

                        if (!ready)
                        {
                            std::unique_lock<std::mutex> lk (lock);
                            cv.wait (lk);
                        }
                    }
                    while (!ready);

                    func ();

                    size_t n_active = active;
                    while (!active.compare_exchange_weak (n_active, n_active - 1)) assert(n_active >= 1);
                    cv.notify_one();
                    *finishedPtr = true;
                }), std::move (finished)));
            }
            cleanup();
        }

        void join() {
            std::unique_lock<std::mutex> lk(lock);
            if (waiting) throw std::logic_error("Already joined");
            waiting = true; // prevent anyone else from modifying the threads (although emplace_back is fine)
            auto thread_iter = threads.begin();
            while (thread_iter != threads.end()) {
                assert(thread_iter->second);
                lk.unlock(); // can't keep the lock because thread might need it
                thread_iter->first.join();
                lk.lock(); // reclaim our lock
                thread_iter = threads.erase(thread_iter);
            }
            waiting = false;
        }

        ~thread_pool() {
            join();
        }
    };
}

#endif //_THREAD_POOL_HPP_
