/** @file queue.c
 * This file contains routines that implement a queue data structure. Currently,
 * this queue is only used by my own coarse grid selection routines.
 *
 * @author David Alber
 * @date January 2005
 */

#include "utilities.h"

void initializeQueue(hypre_Queue * new_queue)
{
  new_queue->head = NULL;
  new_queue->tail = NULL;
  new_queue->length = 0;
}

hypre_Queue * newQueue()
{
  hypre_Queue * new_queue = hypre_TAlloc(hypre_Queue, 1);
  new_queue->head = NULL;
  new_queue->tail = NULL;
  new_queue->length = 0;
  return new_queue;
}

void destroyQueue(hypre_Queue * queue)
{
  // First destroy all of the elements in the queue.
  hypre_QueueElement * next_head;
  while(queue->head) {
    next_head = queue->head->next_elt;
    hypre_TFree(queue->head->data);
    hypre_TFree(queue->head);
    queue->head = next_head;
  }

  // Now free the memory for the queue itself.
  hypre_TFree(queue);
}

void enqueueData(int * data, hypre_Queue * queue)
{
  hypre_QueueElement * new_el = hypre_TAlloc(hypre_QueueElement, 1);

  new_el->data = data;
  new_el->head = 0;
  new_el->tail = 1;
  new_el->own_element = 1;
  new_el->next_elt = NULL;
  new_el->prev_elt = queue->tail;

  if(queue->tail) {
    queue->tail->next_elt = new_el;
    queue->tail->tail = 0;
  }
  else {
    queue->head = new_el;
    new_el->head = 1;
  }

  queue->tail = new_el;
  queue->length++;
}

void enqueueElement(hypre_QueueElement * new_el, hypre_Queue * queue)
{
  new_el->head = 0;
  new_el->tail = 1;
  new_el->own_element = 0;
  new_el->next_elt = NULL;
  new_el->prev_elt = queue->tail;

  if(queue->tail) {
    queue->tail->next_elt = new_el;
    queue->tail->tail = 0;
  }
  else {
    queue->head = new_el;
    new_el->head = 1;
  }

  queue->tail = new_el;
  queue->length++;
}

void pushElement(hypre_QueueElement * new_el, hypre_Queue * queue)
{
  // Put a new element at the head of the queue.
  enqueueElement(new_el, queue);
  moveToHead(new_el, queue);
}

int * dequeue(hypre_Queue * queue)
{
  int * data;
  hypre_QueueElement * prev_head = queue->head;

  if(queue->head == queue->tail)
    queue->tail = NULL;

  if(prev_head) {
    data = queue->head->data;
    // Remove the head element and update the queue accordingly.
    queue->head = queue->head->next_elt;
    if(queue->head) {
      queue->head->prev_elt = NULL;
      queue->head->head = 1;
    }

    if(prev_head->own_element)
      hypre_TFree(prev_head);
    queue->length--;
    return data;
  }
  else
    return NULL;
}

/** removeElement removes the element pointed to by source from the queue. The memory allocated to source is not freed.
 * @param source a hypre_QueueElement pointer pointing to the element to be moved.
 * @param queue a hypre_Queue pointer pointing to the queue in which all of this activity is to take place.
 */
void removeElement(hypre_QueueElement * source, hypre_Queue * queue)
{
  // Update the neighbors of source.
  if(source->prev_elt != NULL) {
    if(source == queue->tail) {
      // Then source is the tail.
      source->prev_elt->tail = 1;
      source->prev_elt->next_elt = NULL;
      queue->tail = source->prev_elt;
      source->tail = 0;
    }
    else {
      // Then point the prev_elt's next pointer at the element after the source
      // and vice versa.
      source->prev_elt->next_elt = source->next_elt;
      source->next_elt->prev_elt = source->prev_elt;
    }
  }

  /////NEW!!!
  else {
    // The source node is the head element.
    queue->head = source->next_elt;
    if(source == queue->tail)
      queue->tail = NULL;
    else {
      queue->head->prev_elt = NULL;
      queue->head->head = 1;
    }
  }
  queue->length--;
}

/** moveAfter moves one element in the queue to a new position in the queue. It is moved immediately in front of the dest element.
 * @param source a hypre_QueueElement pointer pointing to the element to be moved.
 * @param dest a hypre_QueueElement pointer pointing to the element before which the source element is to be moved.
 * @param queue a hypre_Queue pointer pointing to the queue in which all of this activity is to take place.
 */
void moveAfter(hypre_QueueElement * source, hypre_QueueElement * dest, hypre_Queue * queue)
{
  // Make sure that the source is actually going to be moved.
  if(source->prev_elt == dest || source == dest)
    return;

  removeElement(source, queue);

  // Now move the source.
  if(dest == NULL) {
    // Then source is becoming the head.
    source->prev_elt = NULL;
    source->next_elt = queue->head;
    source->head = 1;
    queue->head->prev_elt = source;
    queue->head->head = 0;
    queue->head = source;
  }
  else {
    dest->next_elt->prev_elt = source;
    source->next_elt = dest->next_elt;
    dest->next_elt = source;
    source->prev_elt = dest;
  }
}

/** moveToHead moves the element in the queue pointed to by source to the head of the queue.
 * @param source a hypre_QueueElement pointer pointing to the element to be moved.
 * @param queue a hypre_Queue pointer pointing to the queue in which all of this activity is to take place.
 */
void moveToHead(hypre_QueueElement * source, hypre_Queue * queue)
{
  // Make sure that the source is actually going to be moved.
  if(source == queue->head)
    return;

  removeElement(source, queue);

  // Now move the source to the head of the queue.
  source->prev_elt = NULL;
  source->next_elt = queue->head;
  source->head = 1;

  queue->head->head = 0;
  queue->head->prev_elt = source;

  queue->head = source;
}
