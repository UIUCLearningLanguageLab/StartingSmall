import pyprind


def calc_pp(model, is_test):
    print('Calculating {} perplexity...'.format('test' if is_test else 'train'))
    pp_sum, num_batches, pp = 0, 0, 0
    pbar = pyprind.ProgBar(hub.num_mbs_in_token_ids)
    num_iterations_list = [1] * hub.params.num_parts
    for (x, y) in hub.gen_ids(num_iterations_list, is_test=is_test):
        pbar.update()
        pp_batch = sess.run(graph.batch_pp, feed_dict={graph.x: x, graph.y: y})
        pp_sum += pp_batch
        num_batches += 1
    pp = pp_sum / num_batches
    print('pp={}'.format(pp))
    return pp